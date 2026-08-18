package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// The language server, driven over the wire it actually speaks: Content-Length
// framed JSON-RPC. Testing the handlers directly would skip the framing, and
// the framing is where a language server usually breaks.

// session drives a list of messages through the server and returns everything
// it wrote back, decoded.
func session(t *testing.T, msgs ...map[string]any) []rpcMessage {
	t.Helper()
	var in bytes.Buffer
	for _, m := range msgs {
		m["jsonrpc"] = "2.0"
		body, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&in, "Content-Length: %d\r\n\r\n%s", len(body), body)
	}
	var out bytes.Buffer
	s := &lspServer{out: &out, docs: map[string]string{}}
	r := bufio.NewReader(&in)
	for {
		body, err := readLSPMessage(r)
		if err != nil {
			break
		}
		var msg rpcMessage
		if json.Unmarshal(body, &msg) != nil {
			continue
		}
		if _, done := s.handle(msg); done {
			break
		}
	}
	return decodeAll(t, out.String())
}

func decodeAll(t *testing.T, raw string) []rpcMessage {
	t.Helper()
	var got []rpcMessage
	r := bufio.NewReader(strings.NewReader(raw))
	for {
		body, err := readLSPMessage(r)
		if err != nil {
			break
		}
		var msg rpcMessage
		if err := json.Unmarshal(body, &msg); err != nil {
			t.Fatalf("server wrote something that is not JSON: %s", body)
		}
		got = append(got, msg)
	}
	return got
}

// resultOf re-decodes a reply's result into a map, which is what `any` from a
// round trip through JSON gives.
func resultOf(t *testing.T, m rpcMessage) map[string]any {
	t.Helper()
	b, err := json.Marshal(m.Result)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("result is not an object: %s", b)
	}
	return out
}

func TestLSPInitializeAdvertisesWhatItHas(t *testing.T) {
	got := session(t, map[string]any{"id": 1, "method": "initialize", "params": map[string]any{}})
	if len(got) != 1 {
		t.Fatalf("got %d messages, want 1", len(got))
	}
	caps := resultOf(t, got[0])["capabilities"].(map[string]any)
	for _, want := range []string{"textDocumentSync", "documentFormattingProvider", "hoverProvider"} {
		if _, ok := caps[want]; !ok {
			t.Errorf("capabilities do not advertise %s", want)
		}
	}
	// Completion is deliberately not advertised: roadmap.md's own advice is not
	// to build one before the semantic information is reliable.
	if _, ok := caps["completionProvider"]; ok {
		t.Error("completion is advertised, and there is none")
	}
}

// Opening a file publishes its diagnostics, and editing it republishes them.
func TestLSPPublishesDiagnosticsOnOpenAndChange(t *testing.T) {
	const uri = "file:///m.tw"
	got := session(t,
		map[string]any{"id": 1, "method": "initialize", "params": map[string]any{}},
		map[string]any{"method": "textDocument/didOpen", "params": map[string]any{
			"textDocument": map[string]any{"uri": uri, "text": "let a = zeros(2, 3)\nlet b = zeros(4)\nlet c = a + b\n"},
		}},
		map[string]any{"method": "textDocument/didChange", "params": map[string]any{
			"textDocument":   map[string]any{"uri": uri},
			"contentChanges": []any{map[string]any{"text": "let a = zeros(2, 3)\nlet b = zeros(3)\nlet c = a + b\n"}},
		}},
	)
	var published []rpcMessage
	for _, m := range got {
		if m.Method == "textDocument/publishDiagnostics" {
			published = append(published, m)
		}
	}
	if len(published) != 2 {
		t.Fatalf("got %d publishes, want 2 (open then change)", len(published))
	}

	first := diagsOf(t, published[0])
	if len(first) != 1 {
		t.Fatalf("open published %d diagnostics, want 1: %v", len(first), first)
	}
	msg := first[0]["message"].(string)
	if !strings.Contains(msg, "cannot broadcast") {
		t.Errorf("message = %q, want the broadcast mismatch", msg)
	}
	if line := first[0]["range"].(map[string]any)["start"].(map[string]any)["line"].(float64); line != 2 {
		t.Errorf("diagnostic is on line %v, want 2 (zero-based)", line)
	}

	// The edit fixes the shape, so the republish must be empty rather than
	// absent: an editor clears its squiggles from an empty list.
	if second := diagsOf(t, published[1]); len(second) != 0 {
		t.Errorf("after the fix, %d diagnostics remain: %v", len(second), second)
	}
}

// A file that does not parse publishes the syntax error alone. Reporting an
// unknown name for every identifier after one missing brace is the cascade
// worth not having.
func TestLSPPublishesASyntaxErrorAlone(t *testing.T) {
	got := session(t,
		map[string]any{"method": "textDocument/didOpen", "params": map[string]any{
			"textDocument": map[string]any{"uri": "file:///s.tw", "text": "fn f(x) {\n  x +\n"},
		}},
	)
	diags := diagsOf(t, got[0])
	if len(diags) != 1 {
		t.Fatalf("got %d diagnostics for a syntax error, want exactly 1: %v", len(diags), diags)
	}
}

func TestLSPClosingAFileClearsItsDiagnostics(t *testing.T) {
	const uri = "file:///c.tw"
	got := session(t,
		map[string]any{"method": "textDocument/didOpen", "params": map[string]any{
			"textDocument": map[string]any{"uri": uri, "text": "let c = zeros(2, 3) + zeros(4)\n"},
		}},
		map[string]any{"method": "textDocument/didClose", "params": map[string]any{
			"textDocument": map[string]any{"uri": uri},
		}},
	)
	if len(got) != 2 {
		t.Fatalf("got %d messages, want 2", len(got))
	}
	if n := len(diagsOf(t, got[0])); n != 1 {
		t.Fatalf("open published %d diagnostics, want 1", n)
	}
	if n := len(diagsOf(t, got[1])); n != 0 {
		t.Errorf("close published %d diagnostics, want 0", n)
	}
}

func TestLSPFormatting(t *testing.T) {
	got := session(t,
		map[string]any{"method": "textDocument/didOpen", "params": map[string]any{
			"textDocument": map[string]any{"uri": "file:///f.tw", "text": "let    x=1.0+2.0\n"},
		}},
		map[string]any{"id": 2, "method": "textDocument/formatting", "params": map[string]any{
			"textDocument": map[string]any{"uri": "file:///f.tw"},
		}},
	)
	last := got[len(got)-1]
	b, _ := json.Marshal(last.Result)
	var edits []map[string]any
	if err := json.Unmarshal(b, &edits); err != nil || len(edits) != 1 {
		t.Fatalf("formatting result = %s, want one edit", b)
	}
	if newText := edits[0]["newText"].(string); newText != "let x = 1 + 2\n" {
		t.Errorf("newText = %q", newText)
	}
}

// Hover answers from the checker, so it costs nothing for a value that would
// not fit in memory, and it reports a shape rather than a number.
func TestLSPHover(t *testing.T) {
	const src = "let logits = zeros(32, 128)\nprint(logits)\n"
	got := session(t,
		map[string]any{"method": "textDocument/didOpen", "params": map[string]any{
			"textDocument": map[string]any{"uri": "file:///h.tw", "text": src},
		}},
		map[string]any{"id": 3, "method": "textDocument/hover", "params": map[string]any{
			"textDocument": map[string]any{"uri": "file:///h.tw"},
			// Line 1, inside `logits` in the print call.
			"position": map[string]any{"line": 1, "character": 8},
		}},
	)
	last := got[len(got)-1]
	if last.Result == nil {
		t.Fatal("hover returned nothing")
	}
	contents := resultOf(t, last)["contents"].(map[string]any)
	value := contents["value"].(string)
	for _, want := range []string{"logits", "type", "shape"} {
		if !strings.Contains(value, want) {
			t.Errorf("hover %q does not mention %q", value, want)
		}
	}
}

// Hovering over whitespace, punctuation or a bare number has no answer, and
// answering anyway would put a tooltip over every character in the file.
func TestLSPHoverOnNothing(t *testing.T) {
	for _, pos := range []int{3, 15} { // a space, and inside the digits of `32`
		got := session(t,
			map[string]any{"method": "textDocument/didOpen", "params": map[string]any{
				"textDocument": map[string]any{"uri": "file:///n.tw", "text": "let x = zeros(32, 4)\n"},
			}},
			map[string]any{"id": 4, "method": "textDocument/hover", "params": map[string]any{
				"textDocument": map[string]any{"uri": "file:///n.tw"},
				"position":     map[string]any{"line": 0, "character": pos},
			}},
		)
		if last := got[len(got)-1]; last.Result != nil {
			t.Errorf("hover at character %d answered %v, want nothing", pos, last.Result)
		}
	}
}

// Every request gets a reply, including one the server does not implement, or
// the client waits for it forever.
func TestLSPAnswersAnUnknownRequest(t *testing.T) {
	got := session(t, map[string]any{"id": 9, "method": "textDocument/definition", "params": map[string]any{}})
	if len(got) != 1 {
		t.Fatalf("an unimplemented request got %d replies, want 1", len(got))
	}
	if got[0].Result != nil {
		t.Errorf("result = %v, want null", got[0].Result)
	}
}

// A notification has no id and must not be replied to.
func TestLSPDoesNotReplyToNotifications(t *testing.T) {
	got := session(t, map[string]any{"method": "initialized", "params": map[string]any{}})
	if len(got) != 0 {
		t.Errorf("a notification drew %d replies: %v", len(got), got)
	}
}

func TestLSPWordAt(t *testing.T) {
	const src = "let logits = nn.dense(x, w)\n"
	for _, tc := range []struct {
		char int
		want string
	}{
		{5, "logits"},
		{15, "nn.dense"},
		{22, "x"},
		{3, ""},  // the space after `let`
		{27, ""}, // past the end of the line
	} {
		if got := wordAt(src, 0, tc.char); got != tc.want {
			t.Errorf("wordAt(%d) = %q, want %q", tc.char, got, tc.want)
		}
	}
}

// The protocol's two-step stop: `exit` after `shutdown` is success, `exit`
// without one is a failure, and the specification is specific about it.
func TestLSPExitCodes(t *testing.T) {
	s := &lspServer{out: &bytes.Buffer{}, docs: map[string]string{}}
	if code, done := s.handle(rpcMessage{Method: "exit"}); !done || code != 1 {
		t.Errorf("exit without shutdown = (%d, %v), want (1, true)", code, done)
	}
	s = &lspServer{out: &bytes.Buffer{}, docs: map[string]string{}}
	s.handle(rpcMessage{ID: json.RawMessage("1"), Method: "shutdown"})
	if code, done := s.handle(rpcMessage{Method: "exit"}); !done || code != 0 {
		t.Errorf("exit after shutdown = (%d, %v), want (0, true)", code, done)
	}
}

func diagsOf(t *testing.T, m rpcMessage) []map[string]any {
	t.Helper()
	var p struct {
		Diagnostics []map[string]any `json:"diagnostics"`
	}
	if err := json.Unmarshal(m.Params, &p); err != nil {
		t.Fatalf("params are not a publishDiagnostics: %s", m.Params)
	}
	return p.Diagnostics
}
