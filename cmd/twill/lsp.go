package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/format"
	"github.com/twill-lang/twill/internal/lexer"
	"github.com/twill-lang/twill/internal/parser"
)

// `twill lsp` is a language server: the checker's answers delivered while the
// file is being typed rather than when a command is run.
//
// Three capabilities, and each is an existing answer given a new route rather
// than a second implementation of anything:
//
//   - diagnostics, from checker.Check and the parser, republished on every edit;
//   - formatting, from format.Source, which already refuses rather than move a
//     comment it cannot place;
//   - hover, from checker.Describe, which is what the REPL's `:type` and
//     `:shape` answer.
//
// There is no completion. docs/roadmap.md's own advice is not to implement it
// before the semantic information is reliable, and a completion list built from
// a token scan would be a worse thing that looked like a better one.
//
// The transport is JSON-RPC 2.0 over stdio with LSP's Content-Length framing.
// encoding/json is in Go's standard library, so this costs the zero-dependency
// promise nothing.

type lspServer struct {
	out io.Writer
	mu  sync.Mutex
	// docs is the client's copy of each open file, by URI. The client owns the
	// text once a file is open -- what is on disk may be older -- so everything
	// answered here is answered from this map.
	docs map[string]string
	// shutdownSeen tracks the protocol's two-step exit: `shutdown` then `exit`.
	// An `exit` without a preceding `shutdown` is a failure exit, which the
	// specification is specific about.
	shutdownSeen bool
}

type rpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func runLSP() int {
	s := &lspServer{out: os.Stdout, docs: map[string]string{}}
	r := bufio.NewReader(os.Stdin)
	for {
		body, err := readLSPMessage(r)
		if err != nil {
			// End of input is the editor going away, which is not a failure.
			if err == io.EOF {
				return 0
			}
			fmt.Fprintf(os.Stderr, "twill lsp: %s\n", err)
			return 1
		}
		var msg rpcMessage
		if err := json.Unmarshal(body, &msg); err != nil {
			continue // A message that is not JSON is not answerable.
		}
		if code, done := s.handle(msg); done {
			return code
		}
	}
}

// readLSPMessage reads one Content-Length framed message.
func readLSPMessage(r *bufio.Reader) ([]byte, error) {
	length := -1
	sawHeader := false
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			// The stream ending between messages is the editor going away, not a
			// malformed message, and trailing whitespace after the last one counts
			// as between. Reporting that made a clean shutdown look like a crash.
			if err == io.EOF && !sawHeader && strings.TrimSpace(line) == "" {
				return nil, io.EOF
			}
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break // The blank line ends the header block.
		}
		name, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		sawHeader = true
		if strings.EqualFold(strings.TrimSpace(name), "Content-Length") {
			n, convErr := strconv.Atoi(strings.TrimSpace(value))
			if convErr != nil {
				return nil, fmt.Errorf("bad Content-Length: %q", value)
			}
			length = n
		}
	}
	if length < 0 {
		return nil, fmt.Errorf("message with no Content-Length")
	}
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	return body, nil
}

func (s *lspServer) send(msg rpcMessage) {
	msg.JSONRPC = "2.0"
	body, err := json.Marshal(msg)
	if err != nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(s.out, "Content-Length: %d\r\n\r\n", len(body))
	s.out.Write(body)
}

func (s *lspServer) reply(id json.RawMessage, result any) {
	if id == nil {
		return // A notification has no id and takes no reply.
	}
	s.send(rpcMessage{ID: id, Result: result})
}

// handle answers one message, reporting the process exit code when the client
// has asked the server to stop.
func (s *lspServer) handle(msg rpcMessage) (int, bool) {
	switch msg.Method {
	case "initialize":
		s.reply(msg.ID, map[string]any{
			"capabilities": map[string]any{
				// Full sync: a twill file is a few thousand lines at most and
				// re-checking one is milliseconds, so incremental sync would be
				// bookkeeping bought with nothing.
				"textDocumentSync":           1,
				"documentFormattingProvider": true,
				"hoverProvider":              true,
			},
			"serverInfo": map[string]any{"name": "twill", "version": version},
		})
	case "initialized", "$/setTrace", "workspace/didChangeConfiguration":
		// Nothing to do, and answering an unknown notification with an error is
		// noisier than ignoring it.
	case "shutdown":
		s.shutdownSeen = true
		s.reply(msg.ID, nil)
	case "exit":
		if s.shutdownSeen {
			return 0, true
		}
		return 1, true
	case "textDocument/didOpen":
		var p struct {
			TextDocument struct {
				URI  string `json:"uri"`
				Text string `json:"text"`
			} `json:"textDocument"`
		}
		if json.Unmarshal(msg.Params, &p) == nil {
			s.docs[p.TextDocument.URI] = p.TextDocument.Text
			s.publish(p.TextDocument.URI)
		}
	case "textDocument/didChange":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
			ContentChanges []struct {
				Text string `json:"text"`
			} `json:"contentChanges"`
		}
		if json.Unmarshal(msg.Params, &p) == nil && len(p.ContentChanges) > 0 {
			s.docs[p.TextDocument.URI] = p.ContentChanges[len(p.ContentChanges)-1].Text
			s.publish(p.TextDocument.URI)
		}
	case "textDocument/didSave":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		if json.Unmarshal(msg.Params, &p) == nil {
			s.publish(p.TextDocument.URI)
		}
	case "textDocument/didClose":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		if json.Unmarshal(msg.Params, &p) == nil {
			delete(s.docs, p.TextDocument.URI)
			// A closed file's diagnostics are cleared, or they hang in the
			// problems panel describing a file nobody is looking at.
			s.send(rpcMessage{Method: "textDocument/publishDiagnostics", Params: rawJSON(map[string]any{
				"uri": p.TextDocument.URI, "diagnostics": []any{},
			})})
		}
	case "textDocument/formatting":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		if json.Unmarshal(msg.Params, &p) != nil {
			s.reply(msg.ID, nil)
			break
		}
		src := s.docs[p.TextDocument.URI]
		out, err := format.Source(src)
		if err != nil {
			// The formatter refused, which it does rather than move a comment it
			// cannot place. Refusing an edit is right: the alternative is
			// rewriting the buffer into something the author did not ask for.
			s.reply(msg.ID, nil)
			break
		}
		s.reply(msg.ID, []map[string]any{{
			"range":   wholeDocumentRange(src),
			"newText": out,
		}})
	case "textDocument/hover":
		s.reply(msg.ID, s.hover(msg.Params))
	default:
		// A request the server does not implement still needs an answer, or the
		// client waits for one forever.
		if msg.ID != nil {
			s.reply(msg.ID, nil)
		}
	}
	return 0, false
}

// publish re-checks a document and sends its diagnostics.
//
// A syntax error is published alone: the checker cannot run without a parse,
// and reporting "unknown name" for every identifier in a file with one missing
// brace is the cascade docs/needs.md warns about.
func (s *lspServer) publish(uri string) {
	src := s.docs[uri]
	var diags []map[string]any

	prog, err := parser.Parse(src)
	if err != nil {
		line, col := 0, 0
		if se, ok := err.(*lexer.SyntaxError); ok {
			line, col = se.Line-1, se.Col-1
		}
		if line < 0 {
			line = 0
		}
		if col < 0 {
			col = 0
		}
		diags = append(diags, map[string]any{
			"range":    lineRange(src, line, col),
			"severity": 1,
			"source":   "twill",
			"message":  strings.TrimPrefix(err.Error(), fmt.Sprintf("line %d:%d: ", line+1, col+1)),
		})
	} else {
		for _, d := range checker.Check(prog) {
			diags = append(diags, map[string]any{
				"range":    lineRange(src, d.Line-1, 0),
				"severity": 1,
				"source":   "twill",
				"message":  d.Msg,
			})
		}
	}
	if diags == nil {
		diags = []map[string]any{}
	}
	s.send(rpcMessage{Method: "textDocument/publishDiagnostics", Params: rawJSON(map[string]any{
		"uri": uri, "diagnostics": diags,
	})})
}

// hover answers with the type and shape of the identifier or expression under
// the cursor. The word under the cursor is described on its own, which is what
// makes the answer cheap and what makes it honest: it is what the checker can
// prove about that text, not a claim about a value nobody has computed.
func (s *lspServer) hover(params json.RawMessage) any {
	var p struct {
		TextDocument struct {
			URI string `json:"uri"`
		} `json:"textDocument"`
		Position struct {
			Line      int `json:"line"`
			Character int `json:"character"`
		} `json:"position"`
	}
	if json.Unmarshal(params, &p) != nil {
		return nil
	}
	src := s.docs[p.TextDocument.URI]
	word := wordAt(src, p.Position.Line, p.Position.Character)
	if word == "" {
		return nil
	}
	// Described against the file it sits in, so a local has the type the file
	// gives it. The rest of the document is read for its bindings only.
	desc, err := checker.DescribeIn(src, word)
	if err != nil {
		return nil
	}
	body := "```twill\n" + word + "\n```\n\n**type** " + desc.Type
	if desc.Shape != desc.Type {
		body += "  \n**shape** `" + desc.Shape + "`"
	}
	return map[string]any{
		"contents": map[string]any{"kind": "markdown", "value": body},
	}
}

// wordAt reads the identifier (or dotted path, so `nn.dense` is one word) that
// the given position sits inside.
func wordAt(src string, line, char int) string {
	lines := strings.Split(src, "\n")
	if line < 0 || line >= len(lines) {
		return ""
	}
	text := lines[line]
	if char < 0 || char > len(text) {
		return ""
	}
	isWord := func(b byte) bool {
		return b == '_' || b == '.' || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
	}
	// The cursor has to be on a word. Expanding out of a space would describe
	// whichever token happened to be to the left, which puts a tooltip in every
	// gap between everything.
	if char >= len(text) || !isWord(text[char]) {
		return ""
	}
	start := char
	for start > 0 && isWord(text[start-1]) {
		start--
	}
	end := char
	for end < len(text) && isWord(text[end]) {
		end++
	}
	word := strings.Trim(text[start:end], ".")
	// A bare number is not worth an answer, and neither is nothing.
	if word == "" || (word[0] >= '0' && word[0] <= '9') {
		return ""
	}
	return word
}

func lineRange(src string, line, col int) map[string]any {
	lines := strings.Split(src, "\n")
	end := col + 1
	if line >= 0 && line < len(lines) {
		end = len(lines[line])
		if end <= col {
			end = col + 1
		}
	}
	return map[string]any{
		"start": map[string]any{"line": line, "character": col},
		"end":   map[string]any{"line": line, "character": end},
	}
}

func wholeDocumentRange(src string) map[string]any {
	lines := strings.Split(src, "\n")
	return map[string]any{
		"start": map[string]any{"line": 0, "character": 0},
		"end":   map[string]any{"line": len(lines), "character": 0},
	}
}

func rawJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage("null")
	}
	return b
}
