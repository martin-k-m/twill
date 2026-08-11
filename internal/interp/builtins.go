package interp

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/martin-k-m/twill/internal/gbm"
	"github.com/martin-k-m/twill/internal/tensor"
	"github.com/martin-k-m/twill/internal/value"
)

func (ip *Interp) installBuiltins() {
	def := func(name string, arity int, variadic bool, fn func([]value.Value) (value.Value, error)) {
		ip.Global.Define(name, &value.Builtin{Name: name, Arity: arity, Variadic: variadic, Fn: fn})
	}

	// I/O.
	def("print", -1, true, func(args []value.Value) (value.Value, error) {
		parts := make([]string, len(args))
		for i, a := range args {
			parts[i] = value.Format(a)
		}
		ip.out(strings.Join(parts, " "))
		return value.TheUnit, nil
	})

	// Unary differentiable math.
	unaryOp := func(name string, f func(*tensor.Tensor) *tensor.Tensor) {
		def(name, 1, false, func(a []value.Value) (value.Value, error) {
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			return f(t), nil
		})
	}
	unaryOp("relu", tensor.Relu)
	unaryOp("exp", tensor.Exp)
	unaryOp("log", tensor.Log)
	unaryOp("sin", tensor.Sin)
	unaryOp("cos", tensor.Cos)
	unaryOp("tanh", tensor.Tanh)
	unaryOp("sigmoid", tensor.Sigmoid)
	unaryOp("sqrt", tensor.Sqrt)
	unaryOp("square", tensor.Square)

	def("abs", 1, false, func(a []value.Value) (value.Value, error) {
		t, err := asTensor(a[0], "abs")
		if err != nil {
			return nil, err
		}
		// |x| = relu(x) + relu(-x), keeping it differentiable.
		pos := tensor.Relu(t)
		neg := tensor.Relu(tensor.Neg(t))
		return tensor.Add(pos, neg)
	})

	def("pow", 2, false, func(a []value.Value) (value.Value, error) {
		base, err := asTensor(a[0], "pow")
		if err != nil {
			return nil, err
		}
		p, err := scalarOf(a[1], "pow")
		if err != nil {
			return nil, err
		}
		return tensor.PowScalar(base, p), nil
	})

	// Bitwise ops on I64, operating on scalar integers. The values are float64,
	// so each operand is taken as an int64 and the result handed back as one.
	// `and` and `or` share the boolean keywords' spelling but are the bitwise
	// builtins when called; `not` stays the boolean operator and bitwise
	// complement is spelled `bnot`. `shr` is arithmetic (sign-extending), so it
	// is defined on a negative operand.
	bitOp := func(name string, f func(x, y int64) int64) {
		def(name, 2, false, func(a []value.Value) (value.Value, error) {
			x, err := scalarOf(a[0], name)
			if err != nil {
				return nil, err
			}
			y, err := scalarOf(a[1], name)
			if err != nil {
				return nil, err
			}
			return tensor.Scalar(float64(f(int64(x), int64(y)))), nil
		})
	}
	bitOp("and", func(x, y int64) int64 { return x & y })
	bitOp("or", func(x, y int64) int64 { return x | y })
	bitOp("xor", func(x, y int64) int64 { return x ^ y })
	// Shift counts are masked to 0..63, per docs/language-guide.md, so a shift is
	// always defined rather than depending on the host's out-of-range behaviour.
	bitOp("shl", func(x, y int64) int64 { return x << uint64(y&63) })
	bitOp("shr", func(x, y int64) int64 { return x >> uint64(y&63) })

	def("bnot", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "bnot")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(float64(^int64(x))), nil
	})

	// Res and Opt are built in: their cases are constructors and values in every
	// scope, so `Ok(x)`, `Err(e)`, `Some(x)` and `None` work without a
	// declaration, and postfix `?` unwraps `Ok`/`Some` or returns `Err`/`None`.
	variantCtor := func(name string) {
		def(name, 1, false, func(a []value.Value) (value.Value, error) {
			return &value.Variant{Name: name, Payload: a[0], HasPayload: true}, nil
		})
	}
	variantCtor("Ok")
	variantCtor("Err")
	variantCtor("Some")
	ip.Global.Define("None", &value.Variant{Name: "None"})

	// `unit` names the Unit value, so a systems-mode arm like `None => unit` and
	// any expression that yields nothing has a spelling. `unit` is not reserved,
	// so it stays a field name elsewhere; only the bare name resolves here.
	ip.Global.Define("unit", value.TheUnit)

	// Scalar f64 math for the systems dialect. The tensor ops (sqrt, exp, ...)
	// work on whole tensors; these are the one-scalar-in, one-scalar-out forms
	// the self-hosted sources call, and a scalar is a rank-0 tensor.
	f64op := func(name string, f func(float64) float64) {
		def(name, 1, false, func(a []value.Value) (value.Value, error) {
			x, err := scalarOf(a[0], name)
			if err != nil {
				return nil, err
			}
			return tensor.Scalar(f(x)), nil
		})
	}
	f64op("f64_sqrt", math.Sqrt)
	f64op("f64_exp", math.Exp)
	f64op("f64_log", math.Log)
	f64op("f64_sin", math.Sin)
	f64op("f64_cos", math.Cos)
	f64op("f64_floor", math.Floor)
	f64op("f64_trunc", math.Trunc)
	f64op("f64_ceil", math.Ceil)
	f64op("f64_round", math.Round)
	f64op("f64_tanh", math.Tanh)
	def("f64_mod", 2, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_mod")
		if err != nil {
			return nil, err
		}
		y, err := scalarOf(a[1], "f64_mod")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(math.Mod(x, y)), nil
	})
	// i64 and f64 are the conversion casts: i64 truncates toward zero, f64 is the
	// identity, since a value is already a float64.
	f64op("i64", math.Trunc)
	f64op("f64", func(x float64) float64 { return x })
	def("f64_pow", 2, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_pow")
		if err != nil {
			return nil, err
		}
		y, err := scalarOf(a[1], "f64_pow")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(math.Pow(x, y)), nil
	})

	// Conversions. Values are float64, so f64_of_i64 is the identity and
	// i64_of_f64 truncates toward zero, matching integer conversion.
	def("f64_of_i64", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_of_i64")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(x), nil
	})
	def("i64_of_f64", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "i64_of_f64")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(math.Trunc(x)), nil
	})

	// IEEE bit access, for the float printer and parser. Above 2^53 the integer
	// side loses precision, the same limit the bitwise ops have, since a value is
	// a float64; within that range the round-trip is exact.
	def("f64_bits", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_bits")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(float64(int64(math.Float64bits(x)))), nil
	})
	def("f64_from_bits", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_from_bits")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(math.Float64frombits(uint64(int64(x)))), nil
	})
	def("f64_signbit", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_signbit")
		if err != nil {
			return nil, err
		}
		if math.Signbit(x) {
			return value.Bool(true), nil
		}
		return value.Bool(false), nil
	})

	// --- systems collections: growable list, ordered dict, byte buffer -------

	def("arr_new", 0, false, func(a []value.Value) (value.Value, error) {
		return &value.List{}, nil
	})
	def("push", 2, false, func(a []value.Value) (value.Value, error) {
		l, ok := a[0].(*value.List)
		if !ok {
			return nil, fmt.Errorf("push expects a list")
		}
		l.Items = append(l.Items, a[1])
		return value.TheUnit, nil
	})
	def("pop", 1, false, func(a []value.Value) (value.Value, error) {
		l, ok := a[0].(*value.List)
		if !ok {
			return nil, fmt.Errorf("pop expects a list")
		}
		if len(l.Items) == 0 {
			return nil, fmt.Errorf("pop from an empty list")
		}
		last := l.Items[len(l.Items)-1]
		l.Items = l.Items[:len(l.Items)-1]
		return last, nil
	})
	// arr is a list literal as a call: arr(a, b, c) is [a, b, c]. A fresh list
	// every call, so two callers never share backing storage.
	def("arr", -1, true, func(a []value.Value) (value.Value, error) {
		items := make([]value.Value, len(a))
		copy(items, a)
		return &value.List{Items: items}, nil
	})
	// arr_clear empties a list in place, so a buffer can be reused across passes
	// without reallocating.
	def("arr_clear", 1, false, func(a []value.Value) (value.Value, error) {
		l, ok := a[0].(*value.List)
		if !ok {
			return nil, fmt.Errorf("arr_clear expects a list")
		}
		l.Items = l.Items[:0]
		return value.TheUnit, nil
	})
	// chr is a one-byte string with the given byte value. UTF-8 text is built by
	// concatenating the bytes of a rune, which is why this is a byte and not a
	// rune: the terminal code spells "│" as chr(226)+chr(...)+chr(...).
	def("chr", 1, false, func(a []value.Value) (value.Value, error) {
		n, err := scalarOf(a[0], "chr")
		if err != nil {
			return nil, err
		}
		return value.Str(string([]byte{byte(int(n))})), nil
	})
	// slice is a byte substring s[i:j], clamped to the string so an out-of-range
	// index yields the empty overlap rather than a panic. Indices are byte
	// offsets, matching how the lexer and width code index a Str.
	def("slice", 3, false, func(a []value.Value) (value.Value, error) {
		s, ok := a[0].(value.Str)
		if !ok {
			return nil, fmt.Errorf("slice expects a string")
		}
		i, err := scalarOf(a[1], "slice")
		if err != nil {
			return nil, err
		}
		j, err := scalarOf(a[2], "slice")
		if err != nil {
			return nil, err
		}
		lo, hi := int(i), int(j)
		n := len(s)
		if lo < 0 {
			lo = 0
		}
		if hi > n {
			hi = n
		}
		if lo > hi {
			lo = hi
		}
		return value.Str(string(s)[lo:hi]), nil
	})
	// gpu_available reports whether a GPU backend is present. The bootstrap has no
	// device, so it is always false; the self-hosted GPU code branches on it to
	// fall back to the CPU path.
	def("gpu_available", 0, false, func(a []value.Value) (value.Value, error) {
		return value.Bool(false), nil
	})
	// gpu_device_count is how many GPUs the backend sees. The bootstrap has none,
	// so it is zero, and gpu_available() is just this being positive.
	def("gpu_device_count", 0, false, func(a []value.Value) (value.Value, error) {
		return tensor.Scalar(0), nil
	})
	// is_tty_stdout reports whether standard output is a terminal, so the CLI can
	// choose colour and progress animation over plain text when piped to a file.
	def("is_tty_stdout", 0, false, func(a []value.Value) (value.Value, error) {
		info, err := os.Stdout.Stat()
		if err != nil {
			return value.Bool(false), nil
		}
		return value.Bool(info.Mode()&os.ModeCharDevice != 0), nil
	})
	// window_size is the terminal's size as a { cols, rows } record. The bootstrap
	// has no size syscall, so it returns zeros and the caller falls back to 80x24.
	def("window_size", 0, false, func(a []value.Value) (value.Value, error) {
		r := value.NewRecord()
		r.Set("cols", tensor.Scalar(0))
		r.Set("rows", tensor.Scalar(0))
		return r, nil
	})
	// The GPU device intrinsics are the FFI boundary to a compute backend. This
	// build has none, so each fails loudly rather than pretending to run; the
	// self-hosted GPU code reaches them only after gpu_available() returns true,
	// which here it never does. Registered so the device binding type-checks.
	for _, name := range []string{
		"gpu_device_open", "gpu_device_info", "gpu_device_close",
		"gpu_alloc", "gpu_free", "gpu_write", "gpu_read", "gpu_copy",
		"gpu_program_build", "gpu_kernel", "gpu_set_arg_buffer", "gpu_set_arg_local",
		"gpu_launch", "gpu_finish", "gpu_device_info_i64",
		"gpu_set_arg_i64", "gpu_set_arg_f64",
	} {
		n := name
		def(n, -1, true, func(a []value.Value) (value.Value, error) {
			return nil, fmt.Errorf("%s: no GPU backend in this build", n)
		})
	}

	dictKey := func(v value.Value, name string) (string, error) {
		// Dict keys are strings internally, but the systems dialect also has
		// Dict[I64, V] (the checker keys its recursion guard by function id). An
		// integer key maps to its decimal string; a program keeps one key type per
		// dict, so a numeric and a textual key never alias in practice.
		if s, ok := v.(value.Str); ok {
			return string(s), nil
		}
		if n, ok := value.AsNumber(v); ok {
			return strconv.FormatInt(int64(n), 10), nil
		}
		return "", fmt.Errorf("%s expects a string or integer key", name)
	}
	asDict := func(v value.Value, name string) (*value.Dict, error) {
		d, ok := v.(*value.Dict)
		if !ok {
			return nil, fmt.Errorf("%s expects a dict", name)
		}
		return d, nil
	}
	def("dict_new", 0, false, func(a []value.Value) (value.Value, error) {
		return value.NewDict(), nil
	})
	def("dict_set", 3, false, func(a []value.Value) (value.Value, error) {
		d, err := asDict(a[0], "dict_set")
		if err != nil {
			return nil, err
		}
		k, err := dictKey(a[1], "dict_set")
		if err != nil {
			return nil, err
		}
		d.Set(k, a[2])
		return value.TheUnit, nil
	})
	// dict_get returns an Opt: Some(value) when the key is present, None when not.
	def("dict_get", 2, false, func(a []value.Value) (value.Value, error) {
		d, err := asDict(a[0], "dict_get")
		if err != nil {
			return nil, err
		}
		k, err := dictKey(a[1], "dict_get")
		if err != nil {
			return nil, err
		}
		if v, ok := d.Get(k); ok {
			return &value.Variant{Name: "Some", Payload: v, HasPayload: true}, nil
		}
		return &value.Variant{Name: "None"}, nil
	})
	def("dict_has", 2, false, func(a []value.Value) (value.Value, error) {
		d, err := asDict(a[0], "dict_has")
		if err != nil {
			return nil, err
		}
		k, err := dictKey(a[1], "dict_has")
		if err != nil {
			return nil, err
		}
		_, ok := d.Get(k)
		return value.Bool(ok), nil
	})
	// dict_must is dict_get without the Opt: it aborts on a missing key, for the
	// call sites that have already checked the key is present.
	def("dict_must", 2, false, func(a []value.Value) (value.Value, error) {
		d, err := asDict(a[0], "dict_must")
		if err != nil {
			return nil, err
		}
		k, err := dictKey(a[1], "dict_must")
		if err != nil {
			return nil, err
		}
		if v, ok := d.Get(k); ok {
			return v, nil
		}
		return nil, fmt.Errorf("dict_must: no key %q", k)
	})
	// dict_or returns the value at a key, or a default when it is absent.
	def("dict_or", 3, false, func(a []value.Value) (value.Value, error) {
		d, err := asDict(a[0], "dict_or")
		if err != nil {
			return nil, err
		}
		k, err := dictKey(a[1], "dict_or")
		if err != nil {
			return nil, err
		}
		if v, ok := d.Get(k); ok {
			return v, nil
		}
		return a[2], nil
	})
	def("dict_keys", 1, false, func(a []value.Value) (value.Value, error) {
		d, err := asDict(a[0], "dict_keys")
		if err != nil {
			return nil, err
		}
		out := &value.List{Items: make([]value.Value, len(d.Keys))}
		for i, k := range d.Keys {
			out.Items[i] = value.Str(k)
		}
		return out, nil
	})

	def("bytes_new", 0, false, func(a []value.Value) (value.Value, error) {
		return &value.Bytes{}, nil
	})
	def("bytes_push", 2, false, func(a []value.Value) (value.Value, error) {
		b, ok := a[0].(*value.Bytes)
		if !ok {
			return nil, fmt.Errorf("bytes_push expects a byte buffer")
		}
		n, err := scalarOf(a[1], "bytes_push")
		if err != nil {
			return nil, err
		}
		b.Data = append(b.Data, byte(int64(n)))
		return value.TheUnit, nil
	})
	def("bytes_to_str", 1, false, func(a []value.Value) (value.Value, error) {
		b, ok := a[0].(*value.Bytes)
		if !ok {
			return nil, fmt.Errorf("bytes_to_str expects a byte buffer")
		}
		return value.Str(string(b.Data)), nil
	})

	def("dict_del", 2, false, func(a []value.Value) (value.Value, error) {
		d, err := asDict(a[0], "dict_del")
		if err != nil {
			return nil, err
		}
		k, err := dictKey(a[1], "dict_del")
		if err != nil {
			return nil, err
		}
		if _, ok := d.Map[k]; ok {
			delete(d.Map, k)
			for i, key := range d.Keys {
				if key == k {
					d.Keys = append(d.Keys[:i], d.Keys[i+1:]...)
					break
				}
			}
		}
		return value.TheUnit, nil
	})

	// A buf is a fixed-size byte buffer, the packed storage a narrow dtype tensor
	// keeps. It reuses the Bytes value, allocated to length up front.
	def("buf_new", 1, false, func(a []value.Value) (value.Value, error) {
		n, err := scalarOf(a[0], "buf_new")
		if err != nil {
			return nil, err
		}
		return &value.Bytes{Data: make([]byte, int(n))}, nil
	})
	def("buf_get8", 2, false, func(a []value.Value) (value.Value, error) {
		b, ok := a[0].(*value.Bytes)
		if !ok {
			return nil, fmt.Errorf("buf_get8 expects a buffer")
		}
		i, err := scalarOf(a[1], "buf_get8")
		if err != nil {
			return nil, err
		}
		if int(i) < 0 || int(i) >= len(b.Data) {
			return nil, fmt.Errorf("buf_get8 index %d out of range", int(i))
		}
		return tensor.Scalar(float64(b.Data[int(i)])), nil
	})
	def("buf_set8", 3, false, func(a []value.Value) (value.Value, error) {
		b, ok := a[0].(*value.Bytes)
		if !ok {
			return nil, fmt.Errorf("buf_set8 expects a buffer")
		}
		i, err := scalarOf(a[1], "buf_set8")
		if err != nil {
			return nil, err
		}
		v, err := scalarOf(a[2], "buf_set8")
		if err != nil {
			return nil, err
		}
		if int(i) < 0 || int(i) >= len(b.Data) {
			return nil, fmt.Errorf("buf_set8 index %d out of range", int(i))
		}
		b.Data[int(i)] = byte(int64(v))
		return value.TheUnit, nil
	})

	def("buf_len", 1, false, func(a []value.Value) (value.Value, error) {
		b, ok := a[0].(*value.Bytes)
		if !ok {
			return nil, fmt.Errorf("buf_len expects a buffer")
		}
		return tensor.Scalar(float64(len(b.Data))), nil
	})

	// abort ends the program with a message, for the compiler's unreachable
	// branches and invariant checks.
	def("abort", 1, false, func(a []value.Value) (value.Value, error) {
		return nil, fmt.Errorf("abort: %s", value.Format(a[0]))
	})

	// --- systems I/O ---------------------------------------------------------
	//
	// The compiler's front end reads source and writes diagnostics. write_out and
	// write_err print without a trailing newline (the source supplies its own).
	// The fallible ones return a Res: Ok on success, Err with a message.
	asStr := func(v value.Value) (string, bool) {
		switch t := v.(type) {
		case value.Str:
			return string(t), true
		case *value.Bytes:
			return string(t.Data), true
		}
		return "", false
	}
	def("write_out", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("write_out expects a string")
		}
		fmt.Fprint(os.Stdout, s)
		return value.TheUnit, nil
	})
	def("write_err", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("write_err expects a string")
		}
		fmt.Fprint(os.Stderr, s)
		return value.TheUnit, nil
	})
	def("read_file", 1, false, func(a []value.Value) (value.Value, error) {
		path, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("read_file expects a path")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return &value.Variant{Name: "Err", Payload: value.Str(err.Error()), HasPayload: true}, nil
		}
		// Source is text: the systems dialect reads a file into a Str and indexes
		// its bytes, so read_file yields a Str, not a Bytes.
		return &value.Variant{Name: "Ok", Payload: value.Str(string(data)), HasPayload: true}, nil
	})
	def("write_file", 2, false, func(a []value.Value) (value.Value, error) {
		path, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("write_file expects a path")
		}
		content, ok := asStr(a[1])
		if !ok {
			return nil, fmt.Errorf("write_file expects string content")
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			return &value.Variant{Name: "Err", Payload: value.Str(err.Error()), HasPayload: true}, nil
		}
		return &value.Variant{Name: "Ok", Payload: value.TheUnit, HasPayload: true}, nil
	})
	def("list_dir", 1, false, func(a []value.Value) (value.Value, error) {
		path, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("list_dir expects a path")
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return &value.Variant{Name: "Err", Payload: value.Str(err.Error()), HasPayload: true}, nil
		}
		names := &value.List{Items: make([]value.Value, len(entries))}
		for i, e := range entries {
			names.Items[i] = value.Str(e.Name())
		}
		return &value.Variant{Name: "Ok", Payload: names, HasPayload: true}, nil
	})
	// resolve_path makes a path absolute. With two arguments it resolves the
	// second relative to the first's directory. With one it resolves against the
	// running program's directory -- the form the self-hosted evaluator's file
	// builtins use, where the base is implicit, mirroring how the interpreter's
	// own file I/O resolves a relative path.
	def("resolve_path", -1, true, func(a []value.Value) (value.Value, error) {
		switch len(a) {
		case 1:
			rel, ok := asStr(a[0])
			if !ok {
				return nil, fmt.Errorf("resolve_path expects a path")
			}
			return value.Str(ip.resolvePath(rel)), nil
		case 2:
			base, ok := asStr(a[0])
			if !ok {
				return nil, fmt.Errorf("resolve_path expects a base path")
			}
			rel, ok := asStr(a[1])
			if !ok {
				return nil, fmt.Errorf("resolve_path expects a relative path")
			}
			if filepath.IsAbs(rel) {
				return value.Str(rel), nil
			}
			return value.Str(filepath.Join(filepath.Dir(base), rel)), nil
		default:
			return nil, fmt.Errorf("resolve_path expects 1 or 2 arguments, got %d", len(a))
		}
	})
	def("str_quote", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("str_quote expects a string")
		}
		return value.Str(strconv.Quote(s)), nil
	})
	// i64_of_str parses a decimal integer, returning an Opt.
	def("i64_of_str", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("i64_of_str expects a string")
		}
		n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
		if err != nil {
			return &value.Variant{Name: "None"}, nil
		}
		return &value.Variant{Name: "Some", Payload: tensor.Scalar(float64(n)), HasPayload: true}, nil
	})

	// env reads an environment variable as an Opt: Some(value) when set (even
	// empty), None when absent, so a caller can tell "" from unset.
	def("env", 1, false, func(a []value.Value) (value.Value, error) {
		name, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("env expects a string")
		}
		v, ok := os.LookupEnv(name)
		if !ok {
			return &value.Variant{Name: "None"}, nil
		}
		return &value.Variant{Name: "Some", Payload: value.Str(v), HasPayload: true}, nil
	})

	// clock_now_ms is wall-clock time in milliseconds, for the progress and
	// spinner code that measures elapsed time and rates. Nondeterministic by
	// nature; nothing in the compiler proper reads it.
	def("clock_now_ms", 0, false, func(a []value.Value) (value.Value, error) {
		return tensor.Scalar(float64(time.Now().UnixMilli())), nil
	})

	// emit_line writes a string and a newline, the diagnostic printer's unit.
	def("emit_line", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("emit_line expects a string")
		}
		ip.out(s)
		return value.TheUnit, nil
	})

	// str_to_f64 parses a float from its decimal (or hex-float) text, returning an
	// Opt. std/float.f64_of_str delegates to this on the bootstrap: the pure-twill
	// decimal parser it also contains needs exact 64-bit integers to assemble an
	// IEEE pattern, which a float64-backed interpreter cannot provide, so the
	// self-hosted compiler would otherwise read every numeric literal as garbage.
	// Underscores are digit separators and dropped, matching the twill parser.
	def("str_to_f64", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("str_to_f64 expects a string")
		}
		clean := strings.ReplaceAll(s, "_", "")
		f, err := strconv.ParseFloat(clean, 64)
		if err != nil {
			return &value.Variant{Name: "None"}, nil
		}
		return &value.Variant{Name: "Some", Payload: tensor.Scalar(f), HasPayload: true}, nil
	})

	// f64_to_str is the shortest %g decimal of a float, the same call the Go
	// formatter uses (strconv 'g', -1). std/float.f64_shortest delegates to it on
	// the bootstrap: the pure-twill shortest-print, like the parser, assembles its
	// digits with 64-bit integer arithmetic the float64 runtime cannot do exactly.
	def("f64_to_str", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_to_str")
		if err != nil {
			return nil, err
		}
		return value.Str(strconv.FormatFloat(x, 'g', -1, 64)), nil
	})

	// module_source resolves an import path to its source text: a "std/..." path
	// to the embedded standard library, anything else to a file relative to the
	// running program. It is the one primitive the self-hosted evaluator's
	// exec_import needs; the parsing, evaluation and namespace snapshot are twill.
	// None when the module cannot be read (a bad path, a retired .ra extension).
	def("module_source", 1, false, func(a []value.Value) (value.Value, error) {
		path, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("module_source expects a string")
		}
		mod, err := ip.loadModule(path)
		if err != nil {
			return &value.Variant{Name: "None"}, nil
		}
		return &value.Variant{Name: "Some", Payload: value.Str(mod.src), HasPayload: true}, nil
	})

	// f64_hex is the exact hexadecimal-float form (strconv 'x', the same call
	// cmd/twill/dump.go's hexFloat uses), for the canonical dump the differential
	// harness compares bit for bit. The pure-twill hex printer needs exact 64-bit
	// integer arithmetic to lay out the mantissa, so on the bootstrap the self-
	// hosted canonical dump delegates here.
	def("f64_hex", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "f64_hex")
		if err != nil {
			return nil, err
		}
		return value.Str(strconv.FormatFloat(x, 'x', -1, 64)), nil
	})

	// num_to_text is the runtime number rendering `print` and `str` use: an
	// integer without a point, otherwise `%.6f` with trailing zeros trimmed --
	// exactly value.FormatNumber, the Go bootstrap's own printer. std/float's
	// format_number delegates here so a running self-hosted program prints numbers
	// identically instead of through the pure-twill decimal machinery, which needs
	// an exact 64-bit integer the float64 runtime lacks.
	def("num_to_text", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "num_to_text")
		if err != nil {
			return nil, err
		}
		return value.Str(value.FormatNumber(x)), nil
	})

	// A seeded generator, scalar draws over the interpreter's own rng so a `seed`
	// is reproducible across both the tensor ops and these.
	def("rng_seed", 1, false, func(a []value.Value) (value.Value, error) {
		n, err := scalarOf(a[0], "rng_seed")
		if err != nil {
			return nil, err
		}
		ip.rng.Seed(int64(n))
		return value.TheUnit, nil
	})
	def("rng_uniform", 0, false, func(a []value.Value) (value.Value, error) {
		return tensor.Scalar(ip.rng.Float64()), nil
	})
	def("rng_normal", 0, false, func(a []value.Value) (value.Value, error) {
		return tensor.Scalar(ip.rng.NormFloat64()), nil
	})
	def("rng_perm", 1, false, func(a []value.Value) (value.Value, error) {
		n, err := scalarOf(a[0], "rng_perm")
		if err != nil {
			return nil, err
		}
		perm := ip.rng.Perm(int(n))
		out := &value.List{Items: make([]value.Value, len(perm))}
		for i, p := range perm {
			out.Items[i] = tensor.Scalar(float64(p))
		}
		return out, nil
	})

	// is_same is reference identity: true when two values are the same object, so
	// a mutation through one is seen through the other. Every reference value is a
	// pointer and every scalar is comparable, so interface equality is identity.
	def("is_same", 2, false, func(a []value.Value) (value.Value, error) {
		return value.Bool(a[0] == a[1]), nil
	})

	// args is the argument list the program was invoked with, as strings.
	def("args", 0, false, func(a []value.Value) (value.Value, error) {
		out := &value.List{Items: make([]value.Value, len(ip.Args))}
		for i, s := range ip.Args {
			out.Items[i] = value.Str(s)
		}
		return out, nil
	})

	// save_value / load_value persist a value to a file as its printed form. This
	// is the minimal serializer the compiler's cache needs; a value that does not
	// round-trip through its text is not one these are used on.
	def("save_value", 2, false, func(a []value.Value) (value.Value, error) {
		path, ok := asStr(a[1])
		if !ok {
			return nil, fmt.Errorf("save_value expects a path")
		}
		if err := os.WriteFile(path, []byte(value.Format(a[0])), 0o644); err != nil {
			return &value.Variant{Name: "Err", Payload: value.Str(err.Error()), HasPayload: true}, nil
		}
		return &value.Variant{Name: "Ok", Payload: value.TheUnit, HasPayload: true}, nil
	})
	def("load_value", 1, false, func(a []value.Value) (value.Value, error) {
		path, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("load_value expects a path")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return &value.Variant{Name: "Err", Payload: value.Str(err.Error()), HasPayload: true}, nil
		}
		return &value.Variant{Name: "Ok", Payload: value.Str(string(data)), HasPayload: true}, nil
	})

	binTensor := func(name string, f func(a, b *tensor.Tensor) (*tensor.Tensor, error)) {
		def(name, 2, false, func(a []value.Value) (value.Value, error) {
			x, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			y, err := asTensor(a[1], name)
			if err != nil {
				return nil, err
			}
			return f(x, y)
		})
	}
	binTensor("matmul", tensor.MatMul)
	binTensor("dot", tensor.MatMul)
	binTensor("conv2d", tensor.Conv2D)

	// maxpool2d(x, k): non-overlapping k×k max pooling over each channel.
	def("maxpool2d", 2, false, func(a []value.Value) (value.Value, error) {
		x, err := asTensor(a[0], "maxpool2d")
		if err != nil {
			return nil, err
		}
		k, err := intOf(a[1], "maxpool2d")
		if err != nil {
			return nil, err
		}
		return tensor.MaxPool2D(x, k)
	})

	// gather(x, indices): select rows of x (first axis) by an index list or
	// 1-D tensor. Differentiable — the base of embedding lookups and batching.
	def("gather", 2, false, func(a []value.Value) (value.Value, error) {
		x, err := asTensor(a[0], "gather")
		if err != nil {
			return nil, err
		}
		idx, err := indicesOf(a[1], "gather")
		if err != nil {
			return nil, err
		}
		return tensor.Gather(x, idx)
	})
	binTensor("maximum", tensor.Maximum)
	binTensor("minimum", tensor.Minimum)
	binTensor("greater", tensor.Greater)
	binTensor("less", tensor.Less)
	binTensor("greater_equal", tensor.GreaterEqual)
	binTensor("less_equal", tensor.LessEqual)
	binTensor("equal", tensor.EqualOp)

	def("where", 3, false, func(a []value.Value) (value.Value, error) {
		c, err := asTensor(a[0], "where")
		if err != nil {
			return nil, err
		}
		x, err := asTensor(a[1], "where")
		if err != nil {
			return nil, err
		}
		y, err := asTensor(a[2], "where")
		if err != nil {
			return nil, err
		}
		return tensor.Where(c, x, y)
	})

	def("clip", 3, false, func(a []value.Value) (value.Value, error) {
		t, err := asTensor(a[0], "clip")
		if err != nil {
			return nil, err
		}
		lo, err := scalarOf(a[1], "clip")
		if err != nil {
			return nil, err
		}
		hi, err := scalarOf(a[2], "clip")
		if err != nil {
			return nil, err
		}
		return tensor.Clip(t, lo, hi), nil
	})

	// Reductions: one argument reduces everything to a scalar; a second
	// argument reduces a single axis.
	reduce := func(name string, full func(*tensor.Tensor) *tensor.Tensor,
		axis func(*tensor.Tensor, int) (*tensor.Tensor, error)) {
		def(name, -1, true, func(a []value.Value) (value.Value, error) {
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			switch len(a) {
			case 1:
				return full(t), nil
			case 2:
				ax, err := intOf(a[1], name)
				if err != nil {
					return nil, err
				}
				return axis(t, ax)
			default:
				return nil, fmt.Errorf("%s expects (tensor) or (tensor, axis)", name)
			}
		})
	}
	reduce("sum", tensor.Sum, tensor.SumAxis)
	reduce("mean", tensor.Mean, tensor.MeanAxis)
	reduce("max", tensor.MaxAll, tensor.MaxAxis)
	reduce("min", tensor.MinAll, tensor.MinAxis)
	reduce("prod", tensor.Prod, tensor.ProdAxis)
	reduce("median", tensor.Median, tensor.MedianAxis)

	// sort/argsort take an optional axis and an optional "desc" flag, so
	// `sort(t)`, `sort(t, 0)` and `sort(t, 0, true)` all read the way they look.
	sorting := func(name string, run func(*tensor.Tensor, int, bool) (*tensor.Tensor, error)) {
		def(name, -1, true, func(a []value.Value) (value.Value, error) {
			if len(a) == 0 || len(a) > 3 {
				return nil, fmt.Errorf("%s expects (tensor[, axis[, descending]])", name)
			}
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			// Default to the last axis, matching softmax and argmax rather than
			// reducing over everything: sorting a matrix almost always means
			// sorting each row.
			axis := len(t.Shape) - 1
			if len(a) >= 2 {
				if axis, err = intOf(a[1], name); err != nil {
					return nil, err
				}
			}
			descending := false
			if len(a) == 3 {
				d, err := scalarOf(a[2], name)
				if err != nil {
					return nil, err
				}
				descending = d != 0
			}
			return run(t, axis, descending)
		})
	}
	sorting("sort", tensor.SortAxis)
	sorting("argsort", tensor.ArgsortAxis)

	// topk/argtopk need k, so they take (tensor, k[, axis[, smallest]]).
	topping := func(name string, run func(*tensor.Tensor, int, int, bool) (*tensor.Tensor, error)) {
		def(name, -1, true, func(a []value.Value) (value.Value, error) {
			if len(a) < 2 || len(a) > 4 {
				return nil, fmt.Errorf("%s expects (tensor, k[, axis[, smallest]])", name)
			}
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			k, err := intOf(a[1], name)
			if err != nil {
				return nil, err
			}
			axis := len(t.Shape) - 1
			if len(a) >= 3 {
				if axis, err = intOf(a[2], name); err != nil {
					return nil, err
				}
			}
			largest := true
			if len(a) == 4 {
				s, err := scalarOf(a[3], name)
				if err != nil {
					return nil, err
				}
				largest = s == 0
			}
			return run(t, k, axis, largest)
		})
	}
	topping("topk", tensor.TopKAxis)
	topping("argtopk", tensor.ArgTopKAxis)

	// argmax/argmin/flip all take an optional axis and default to the last one.
	lastAxis := func(name string, run func(*tensor.Tensor, int) (*tensor.Tensor, error)) {
		def(name, -1, true, func(a []value.Value) (value.Value, error) {
			if len(a) == 0 || len(a) > 2 {
				return nil, fmt.Errorf("%s expects (tensor[, axis])", name)
			}
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			axis := len(t.Shape) - 1
			if len(a) == 2 {
				if axis, err = intOf(a[1], name); err != nil {
					return nil, err
				}
			}
			return run(t, axis)
		})
	}
	lastAxis("argmax", tensor.ArgmaxAxis)
	lastAxis("argmin", tensor.ArgminAxis)
	lastAxis("flip", tensor.FlipAxis)
	lastAxis("diff", tensor.DiffAxis)

	// roll takes the shift first, since that is the argument nobody omits.
	def("roll", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) < 2 || len(a) > 3 {
			return nil, fmt.Errorf("roll expects (tensor, shift[, axis])")
		}
		t, err := asTensor(a[0], "roll")
		if err != nil {
			return nil, err
		}
		shift, err := intOf(a[1], "roll")
		if err != nil {
			return nil, err
		}
		axis := len(t.Shape) - 1
		if len(a) == 3 {
			if axis, err = intOf(a[2], "roll"); err != nil {
				return nil, err
			}
		}
		return tensor.RollAxis(t, shift, axis)
	})

	// Axis-aware ops that default to the last axis.
	lastAxisOp := func(name string, f func(*tensor.Tensor, int) (*tensor.Tensor, error)) {
		def(name, -1, true, func(a []value.Value) (value.Value, error) {
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			axis := len(t.Shape) - 1
			if len(a) == 2 {
				axis, err = intOf(a[1], name)
				if err != nil {
					return nil, err
				}
			}
			return f(t, axis)
		})
	}
	lastAxisOp("softmax", tensor.Softmax)
	lastAxisOp("logsumexp", tensor.LogSumExp)

	def("reshape", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) < 2 {
			return nil, fmt.Errorf("reshape expects (tensor, ...shape)")
		}
		t, err := asTensor(a[0], "reshape")
		if err != nil {
			return nil, err
		}
		shape, err := shapeFromArgs(a[1:], "reshape")
		if err != nil {
			return nil, err
		}
		return tensor.Reshape(t, shape)
	})

	def("broadcast_to", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) < 2 {
			return nil, fmt.Errorf("broadcast_to expects (tensor, ...shape)")
		}
		t, err := asTensor(a[0], "broadcast_to")
		if err != nil {
			return nil, err
		}
		shape, err := shapeFromArgs(a[1:], "broadcast_to")
		if err != nil {
			return nil, err
		}
		return tensor.BroadcastTo(t, shape)
	})

	def("transpose", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) < 1 {
			return nil, fmt.Errorf("transpose expects a tensor")
		}
		t, err := asTensor(a[0], "transpose")
		if err != nil {
			return nil, err
		}
		axes := make([]int, 0, len(a)-1)
		for _, v := range a[1:] {
			ax, err := intOf(v, "transpose")
			if err != nil {
				return nil, err
			}
			axes = append(axes, ax)
		}
		return tensor.TransposePerm(t, axes)
	})

	def("einsum", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) < 2 {
			return nil, fmt.Errorf("einsum expects a spec string and at least one tensor")
		}
		spec, ok := a[0].(value.Str)
		if !ok {
			return nil, fmt.Errorf("einsum's first argument must be a spec string")
		}
		tensors := make([]*tensor.Tensor, len(a)-1)
		for i, v := range a[1:] {
			t, err := asTensor(v, "einsum")
			if err != nil {
				return nil, err
			}
			tensors[i] = t
		}
		return tensor.Einsum(string(spec), tensors)
	})

	def("concat", 2, false, func(a []value.Value) (value.Value, error) {
		items, err := toItems(a[0], "concat")
		if err != nil {
			return nil, err
		}
		tensors := make([]*tensor.Tensor, len(items))
		for i, it := range items {
			tensors[i], err = asTensor(it, "concat")
			if err != nil {
				return nil, err
			}
		}
		axis, err := intOf(a[1], "concat")
		if err != nil {
			return nil, err
		}
		return tensor.Concat(tensors, axis)
	})

	// split(t, n | sizes[, axis]): the inverse of concat, returning a list.
	// A scalar second argument means that many equal pieces; a list or 1-D
	// tensor means those exact lengths. The axis defaults to 0, matching the
	// axis a bare `concat` result is usually built along.
	def("split", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) < 2 || len(a) > 3 {
			return nil, fmt.Errorf("split expects (tensor, n | sizes[, axis])")
		}
		t, err := asTensor(a[0], "split")
		if err != nil {
			return nil, err
		}
		axis := 0
		if len(a) == 3 {
			if axis, err = intOf(a[2], "split"); err != nil {
				return nil, err
			}
		}

		var pieces []*tensor.Tensor
		// A scalar and a one-element list mean different things — 2 pieces
		// versus one piece of length 2 — so the scalar case is tested first
		// and explicitly, rather than falling through to indicesOf.
		if _, isList := a[1].(*value.List); !isList && isScalarValue(a[1]) {
			n, err := intOf(a[1], "split")
			if err != nil {
				return nil, err
			}
			pieces, err = tensor.SplitEqual(t, n, axis)
			if err != nil {
				return nil, err
			}
		} else {
			sizes, err := indicesOf(a[1], "split")
			if err != nil {
				return nil, err
			}
			pieces, err = tensor.Split(t, sizes, axis)
			if err != nil {
				return nil, err
			}
		}

		items := make([]value.Value, len(pieces))
		for i, p := range pieces {
			items[i] = p
		}
		return &value.List{Items: items}, nil
	})

	// Automatic differentiation.
	def("grad", 1, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		if err := refuseNestedGrad("grad", f); err != nil {
			return nil, err
		}
		return &value.Builtin{Name: "grad(fn)", Variadic: true, Arity: -1, Fn: func(call []value.Value) (value.Value, error) {
			_, grad0, _, err := ip.gradients(f, call)
			return grad0, err
		}}, nil
	})

	def("grads", 1, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		if err := refuseNestedGrad("grads", f); err != nil {
			return nil, err
		}
		return &value.Builtin{Name: "grads(fn)", Variadic: true, Arity: -1, Fn: func(call []value.Value) (value.Value, error) {
			_, _, all, err := ip.gradients(f, call)
			if err != nil {
				return nil, err
			}
			return &value.List{Items: all}, nil
		}}, nil
	})

	def("value_and_grad", 1, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		return &value.Builtin{Name: "value_and_grad(fn)", Variadic: true, Arity: -1, Fn: func(call []value.Value) (value.Value, error) {
			val, grad0, _, err := ip.gradients(f, call)
			if err != nil {
				return nil, err
			}
			return &value.List{Items: []value.Value{val, grad0}}, nil
		}}, nil
	})

	// jacobian(f)(x): the full matrix of partials of a vector output f(x) with
	// respect to the vector input x — one reverse-mode pass per output element.
	def("jacobian", 1, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		return &value.Builtin{Name: "jacobian(fn)", Arity: 1, Fn: func(call []value.Value) (value.Value, error) {
			if len(call) != 1 {
				return nil, fmt.Errorf("jacobian(f) takes exactly one argument")
			}
			return ip.jacobian(f, call[0])
		}}, nil
	})

	// hessian(f)(x): the matrix of second partial derivatives of a scalar f at
	// x, by forward-mode 2-jets (exact second-order autodiff).
	def("hessian", 1, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		return &value.Builtin{Name: "hessian(fn)", Arity: 1, Fn: func(call []value.Value) (value.Value, error) {
			if len(call) != 1 {
				return nil, fmt.Errorf("hessian(f) takes exactly one argument")
			}
			return ip.hessian(f, call[0])
		}}, nil
	})

	// Higher-order list helpers.
	def("map", 2, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		items, err := toItems(a[1], "map")
		if err != nil {
			return nil, err
		}
		out := make([]value.Value, len(items))
		for i, it := range items {
			out[i] = ip.Apply(f, []value.Value{it}, 0)
		}
		return &value.List{Items: out}, nil
	})

	def("zip", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) == 0 {
			return &value.List{}, nil
		}
		cols := make([][]value.Value, len(a))
		minLen := -1
		for i, arg := range a {
			items, err := toItems(arg, "zip")
			if err != nil {
				return nil, err
			}
			cols[i] = items
			if minLen < 0 || len(items) < minLen {
				minLen = len(items)
			}
		}
		out := make([]value.Value, minLen)
		for r := 0; r < minLen; r++ {
			row := make([]value.Value, len(a))
			for c := range a {
				row[c] = cols[c][r]
			}
			out[r] = &value.List{Items: row}
		}
		return &value.List{Items: out}, nil
	})

	// fold(f, init, xs): left fold, acc = f(acc, x) over xs.
	def("fold", 3, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		acc := a[1]
		items, err := toItems(a[2], "fold")
		if err != nil {
			return nil, err
		}
		for _, it := range items {
			acc = ip.Apply(f, []value.Value{acc, it}, 0)
		}
		return acc, nil
	})

	def("append", 2, false, func(a []value.Value) (value.Value, error) {
		items, err := toItems(a[0], "append")
		if err != nil {
			return nil, err
		}
		out := make([]value.Value, 0, len(items)+1)
		out = append(out, items...)
		out = append(out, a[1])
		return &value.List{Items: out}, nil
	})

	def("enumerate", 1, false, func(a []value.Value) (value.Value, error) {
		items, err := toItems(a[0], "enumerate")
		if err != nil {
			return nil, err
		}
		out := make([]value.Value, len(items))
		for i, it := range items {
			out[i] = &value.List{Items: []value.Value{tensor.Scalar(float64(i)), it}}
		}
		return &value.List{Items: out}, nil
	})

	// map_leaves(f, tree): apply f to every tensor leaf of a tree (a tensor, or
	// a list/record nesting tensors), preserving the structure.
	def("map_leaves", 2, false, func(a []value.Value) (value.Value, error) {
		return ip.mapLeaves(a[0], a[1]), nil
	})

	// zip_leaves(f, trees): given a list of same-shaped trees, call f with the
	// list of leaves at each position, preserving the structure.
	def("zip_leaves", 2, false, func(a []value.Value) (value.Value, error) {
		trees, ok := a[1].(*value.List)
		if !ok {
			return nil, fmt.Errorf("zip_leaves expects a list of trees as its second argument")
		}
		return ip.zipLeaves(a[0], trees.Items), nil
	})

	// Tensor construction.
	def("tensor", 1, false, func(a []value.Value) (value.Value, error) {
		if t, ok := a[0].(*tensor.Tensor); ok {
			return t, nil
		}
		nested, err := valueToNested(a[0])
		if err != nil {
			return nil, err
		}
		return tensor.FromNested(nested)
	})
	def("scalar", 1, false, func(a []value.Value) (value.Value, error) {
		x, err := scalarOf(a[0], "scalar")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(x), nil
	})
	def("zeros", -1, true, func(a []value.Value) (value.Value, error) {
		shape, err := shapeFromArgs(a, "zeros")
		if err != nil {
			return nil, err
		}
		return tensor.Filled(shape, 0), nil
	})
	def("ones", -1, true, func(a []value.Value) (value.Value, error) {
		shape, err := shapeFromArgs(a, "ones")
		if err != nil {
			return nil, err
		}
		return tensor.Filled(shape, 1), nil
	})
	def("fill", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) < 1 {
			return nil, fmt.Errorf("fill expects (value, ...shape)")
		}
		v, err := scalarOf(a[0], "fill")
		if err != nil {
			return nil, err
		}
		shape, err := shapeFromArgs(a[1:], "fill")
		if err != nil {
			return nil, err
		}
		return tensor.Filled(shape, v), nil
	})
	def("randn", -1, true, func(a []value.Value) (value.Value, error) {
		shape, err := shapeFromArgs(a, "randn")
		if err != nil {
			return nil, err
		}
		return randomTensor(shape, ip.rng.NormFloat64), nil
	})
	def("rand", -1, true, func(a []value.Value) (value.Value, error) {
		shape, err := shapeFromArgs(a, "rand")
		if err != nil {
			return nil, err
		}
		return randomTensor(shape, ip.rng.Float64), nil
	})
	// seed makes subsequent randn/rand reproducible from a chosen starting point.
	def("seed", 1, false, func(a []value.Value) (value.Value, error) {
		n, err := intOf(a[0], "seed")
		if err != nil {
			return nil, err
		}
		ip.rng.Seed(int64(n))
		return value.TheUnit, nil
	})
	// permutation(n): a shuffled ordering of 0..n-1, as a list, from the seeded
	// RNG (so it's reproducible). Use it with gather to shuffle a dataset.
	def("permutation", 1, false, func(a []value.Value) (value.Value, error) {
		n, err := intOf(a[0], "permutation")
		if err != nil {
			return nil, err
		}
		if n < 0 {
			return nil, fmt.Errorf("permutation: n must be >= 0")
		}
		perm := ip.rng.Perm(n)
		items := make([]value.Value, n)
		for i, p := range perm {
			items[i] = tensor.Scalar(float64(p))
		}
		return &value.List{Items: items}, nil
	})
	def("eye", 1, false, func(a []value.Value) (value.Value, error) {
		n, err := intOf(a[0], "eye")
		if err != nil {
			return nil, err
		}
		d := make([]float64, n*n)
		for i := 0; i < n; i++ {
			d[i*n+i] = 1
		}
		return tensor.New(d, []int{n, n}), nil
	})
	// Inspection and utilities.
	def("shape", 1, false, func(a []value.Value) (value.Value, error) {
		t, err := asTensor(a[0], "shape")
		if err != nil {
			return nil, err
		}
		items := make([]value.Value, len(t.Shape))
		for i, d := range t.Shape {
			items[i] = tensor.Scalar(float64(d))
		}
		return &value.List{Items: items}, nil
	})
	def("len", 1, false, func(a []value.Value) (value.Value, error) {
		switch t := a[0].(type) {
		case *tensor.Tensor:
			if len(t.Shape) == 0 {
				return tensor.Scalar(1), nil
			}
			return tensor.Scalar(float64(t.Shape[0])), nil
		case *value.List:
			return tensor.Scalar(float64(len(t.Items))), nil
		case value.Str:
			// A Str is a byte string, so its length is its byte count, matching how
			// the lexer and the systems string code index it.
			return tensor.Scalar(float64(len(t))), nil
		case *value.Dict:
			return tensor.Scalar(float64(len(t.Keys))), nil
		case *value.Bytes:
			return tensor.Scalar(float64(len(t.Data))), nil
		}
		return nil, fmt.Errorf("len expects a tensor, list, string, dict or bytes")
	})
	def("item", 1, false, func(a []value.Value) (value.Value, error) {
		t, err := asTensor(a[0], "item")
		if err != nil {
			return nil, err
		}
		if t.Size() != 1 {
			return nil, fmt.Errorf("item expects a single-element tensor")
		}
		return tensor.Scalar(t.Data[0]), nil
	})
	def("range", -1, true, func(a []value.Value) (value.Value, error) {
		start, end, step := 0, 0, 1
		var err error
		switch len(a) {
		case 1:
			end, err = intOf(a[0], "range")
		case 2:
			start, err = intOf(a[0], "range")
			if err == nil {
				end, err = intOf(a[1], "range")
			}
		case 3:
			start, err = intOf(a[0], "range")
			if err == nil {
				end, err = intOf(a[1], "range")
			}
			if err == nil {
				step, err = intOf(a[2], "range")
			}
		default:
			return nil, fmt.Errorf("range expects 1-3 arguments")
		}
		if err != nil {
			return nil, err
		}
		if step == 0 {
			return nil, fmt.Errorf("range step cannot be 0")
		}
		var items []value.Value
		for x := start; (step > 0 && x < end) || (step < 0 && x > end); x += step {
			items = append(items, tensor.Scalar(float64(x)))
		}
		return &value.List{Items: items}, nil
	})
	def("list", -1, true, func(a []value.Value) (value.Value, error) {
		items := make([]value.Value, len(a))
		copy(items, a)
		return &value.List{Items: items}, nil
	})
	def("str", 1, false, func(a []value.Value) (value.Value, error) {
		return value.Str(value.Format(a[0])), nil
	})
	// int(x): truncate a scalar toward zero (handy for computing sizes/indices).
	def("int", 1, false, func(a []value.Value) (value.Value, error) {
		f, err := scalarOf(a[0], "int")
		if err != nil {
			return nil, err
		}
		return tensor.Scalar(math.Trunc(f)), nil
	})

	// read_csv(path): load a file of numeric rows (comma- or whitespace-
	// separated) into a [rows, cols] tensor. Blank and '#' lines are skipped.
	def("read_csv", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := a[0].(value.Str)
		if !ok {
			return nil, fmt.Errorf("read_csv expects a string path")
		}
		return ip.readCSV(string(s))
	})

	// read_frame(path): a CSV whose first row is a header, returned as a record
	// mapping each column name to its column tensor (a "frame").
	def("read_frame", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := a[0].(value.Str)
		if !ok {
			return nil, fmt.Errorf("read_frame expects a string path")
		}
		return ip.readFrame(string(s))
	})

	// write_frame(frame, path): write a record of equal-length column tensors to
	// a CSV with a header.
	def("write_frame", 2, false, func(a []value.Value) (value.Value, error) {
		rec, ok := a[0].(*value.Record)
		if !ok {
			return nil, fmt.Errorf("write_frame expects a frame (record of columns)")
		}
		path, ok := a[1].(value.Str)
		if !ok {
			return nil, fmt.Errorf("write_frame expects a string path")
		}
		return value.TheUnit, ip.writeFrame(rec, string(path))
	})

	// save(value, path): write any value (tensors, records, lists, or a fitted
	// gbm model) to a file, so a trained model can be persisted and shipped.
	def("save", 2, false, func(a []value.Value) (value.Value, error) {
		path, ok := a[1].(value.Str)
		if !ok {
			return nil, fmt.Errorf("save expects a string path as its second argument")
		}
		if err := saveValue(a[0], ip.resolvePath(string(path))); err != nil {
			return nil, fmt.Errorf("save: %v", err)
		}
		return value.TheUnit, nil
	})

	// load(path): read back a value written by save.
	def("load", 1, false, func(a []value.Value) (value.Value, error) {
		path, ok := a[0].(value.Str)
		if !ok {
			return nil, fmt.Errorf("load expects a string path")
		}
		v, err := loadValue(ip.resolvePath(string(path)))
		if err != nil {
			return nil, fmt.Errorf("load: %v", err)
		}
		return v, nil
	})

	// columns(record): the field names, in order, as a list of strings.
	def("columns", 1, false, func(a []value.Value) (value.Value, error) {
		rec, ok := a[0].(*value.Record)
		if !ok {
			return nil, fmt.Errorf("columns expects a record")
		}
		items := make([]value.Value, len(rec.Keys))
		for i, k := range rec.Keys {
			items[i] = value.Str(k)
		}
		return &value.List{Items: items}, nil
	})

	// field(record, name): look up a field by string name (dynamic access).
	def("field", 2, false, func(a []value.Value) (value.Value, error) {
		rec, ok := a[0].(*value.Record)
		if !ok {
			return nil, fmt.Errorf("field expects a record")
		}
		name, ok := a[1].(value.Str)
		if !ok {
			return nil, fmt.Errorf("field expects a string name")
		}
		v, present := rec.Get(string(name))
		if !present {
			return nil, fmt.Errorf("record has no field %q", string(name))
		}
		return v, nil
	})

	// with_field(record, name, value): a copy of the record with the field set.
	def("with_field", 3, false, func(a []value.Value) (value.Value, error) {
		rec, ok := a[0].(*value.Record)
		if !ok {
			return nil, fmt.Errorf("with_field expects a record")
		}
		name, ok := a[1].(value.Str)
		if !ok {
			return nil, fmt.Errorf("with_field expects a string name")
		}
		out := value.NewRecord()
		for _, k := range rec.Keys {
			out.Set(k, rec.Fields[k])
		}
		out.Set(string(name), a[2])
		return out, nil
	})

	// Cumulative scans over a sequence (a 1-D tensor's elements in order): each
	// output element folds in the next input. Used to build signals, equity
	// curves, and running peaks for backtests. Differentiable: the scan is a
	// chain of adds, multiplies, or selections, so reverse mode unrolls it.
	// With an axis they scan along it instead, keeping the shape. The split
	// follows the reductions: `sum(t)` covers everything and `sum(t, 0)` works
	// per axis, so `cumsum(t)` and `cumsum(t, 0)` should mean the matching pair.
	// On a 1-D tensor, which is what a sequence is, the two are the same thing.
	scan := func(name string,
		flat func(*tensor.Tensor) *tensor.Tensor,
		along func(*tensor.Tensor, int) (*tensor.Tensor, error),
	) {
		def(name, -1, true, func(a []value.Value) (value.Value, error) {
			if len(a) == 0 || len(a) > 2 {
				return nil, fmt.Errorf("%s expects (tensor[, axis])", name)
			}
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			if len(a) == 1 {
				return flat(t), nil
			}
			axis, err := intOf(a[1], name)
			if err != nil {
				return nil, err
			}
			return along(t, axis)
		})
	}
	scan("cumsum", tensor.CumSum, tensor.CumsumAxis)
	scan("cumprod", tensor.CumProd, tensor.CumprodAxis)
	scan("cummax", tensor.CumMax, tensor.CumMaxAxis)
	scan("cummin", tensor.CumMin, tensor.CumMinAxis)

	// Elementwise rounding (forward-only), handy for turning random draws into
	// integer indices/tokens.
	elemOp := func(name string, f func(float64) float64) {
		def(name, 1, false, func(a []value.Value) (value.Value, error) {
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			out := make([]float64, len(t.Data))
			for i, v := range t.Data {
				out[i] = f(v)
			}
			return tensor.New(out, append([]int{}, t.Shape...)), nil
		})
	}
	elemOp("floor", math.Floor)
	elemOp("ceil", math.Ceil)
	elemOp("round", math.Round)

	// gbm_fit(X, y) or gbm_fit(X, y, opts): train gradient-boosted trees on a
	// [n, d] feature matrix and an [n] target/label vector. opts is an optional
	// record of hyperparameters (rounds, learning_rate, max_depth, min_leaf,
	// lambda, gamma, objective). Returns an opaque model.
	def("gbm_fit", -1, true, func(a []value.Value) (value.Value, error) {
		if len(a) != 2 && len(a) != 3 {
			return nil, fmt.Errorf("gbm_fit expects (X, y) or (X, y, opts)")
		}
		X, err := asTensor(a[0], "gbm_fit")
		if err != nil {
			return nil, err
		}
		if len(X.Shape) != 2 {
			return nil, fmt.Errorf("gbm_fit: X must be a 2-D [n, d] tensor, got shape %v", X.Shape)
		}
		y, err := asTensor(a[1], "gbm_fit")
		if err != nil {
			return nil, err
		}
		if len(y.Shape) != 1 {
			return nil, fmt.Errorf("gbm_fit: y must be a 1-D [n] tensor, got shape %v", y.Shape)
		}
		n, d := X.Shape[0], X.Shape[1]
		if y.Shape[0] != n {
			return nil, fmt.Errorf("gbm_fit: X has %d rows but y has %d", n, y.Shape[0])
		}
		p := gbm.DefaultParams()
		if len(a) == 3 {
			opts, ok := a[2].(*value.Record)
			if !ok {
				return nil, fmt.Errorf("gbm_fit: opts must be a record")
			}
			if err := gbmOptsFromRecord(opts, &p); err != nil {
				return nil, err
			}
		}
		return gbm.Fit(X.Data, y.Data, n, d, p)
	})

	// gbm_predict(model, X): score a [n, d] feature matrix with a fitted model,
	// returning an [n] tensor (raw scores for regression, probabilities for a
	// logistic model).
	def("gbm_predict", 2, false, func(a []value.Value) (value.Value, error) {
		m, ok := a[0].(*gbm.Model)
		if !ok {
			return nil, fmt.Errorf("gbm_predict: first argument must be a model from gbm_fit")
		}
		X, err := asTensor(a[1], "gbm_predict")
		if err != nil {
			return nil, err
		}
		if len(X.Shape) != 2 {
			return nil, fmt.Errorf("gbm_predict: X must be a 2-D [n, d] tensor, got shape %v", X.Shape)
		}
		out, err := m.Predict(X.Data, X.Shape[0], X.Shape[1])
		if err != nil {
			return nil, err
		}
		return tensor.New(out, []int{X.Shape[0]}), nil
	})
}

// gbmOptsFromRecord overrides GBM parameters from an opts record. Unknown keys
// are ignored so the option set can grow without breaking callers.
func gbmOptsFromRecord(opts *value.Record, p *gbm.Params) error {
	getFloat := func(key string) (float64, bool, error) {
		v, ok := opts.Get(key)
		if !ok {
			return 0, false, nil
		}
		f, err := scalarOf(v, "gbm_fit option "+key)
		return f, true, err
	}
	if f, ok, err := getFloat("rounds"); err != nil {
		return err
	} else if ok {
		p.Rounds = int(f)
	}
	if f, ok, err := getFloat("max_depth"); err != nil {
		return err
	} else if ok {
		p.MaxDepth = int(f)
	}
	if f, ok, err := getFloat("min_leaf"); err != nil {
		return err
	} else if ok {
		p.MinLeaf = int(f)
	}
	if f, ok, err := getFloat("learning_rate"); err != nil {
		return err
	} else if ok {
		p.LearningRate = f
	}
	if f, ok, err := getFloat("lambda"); err != nil {
		return err
	} else if ok {
		p.Lambda = f
	}
	if f, ok, err := getFloat("gamma"); err != nil {
		return err
	} else if ok {
		p.Gamma = f
	}
	if v, ok := opts.Get("objective"); ok {
		s, ok := v.(value.Str)
		if !ok {
			return fmt.Errorf("gbm_fit: objective must be a string")
		}
		switch string(s) {
		case "squared":
			p.Objective = gbm.Squared
		case "logistic":
			p.Objective = gbm.Logistic
		default:
			return fmt.Errorf("gbm_fit: unknown objective %q (use \"squared\" or \"logistic\")", s)
		}
	}
	return nil
}

func (ip *Interp) readCSV(path string) (value.Value, error) {
	if !filepath.IsAbs(path) {
		path = filepath.Join(ip.currentDir(), path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read_csv: cannot read %q", path)
	}
	var rows [][]float64
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.FieldsFunc(line, func(r rune) bool {
			return r == ',' || r == ' ' || r == '\t' || r == '\r' || r == ';'
		})
		row := make([]float64, 0, len(fields))
		for _, f := range fields {
			v, perr := strconv.ParseFloat(f, 64)
			if perr != nil {
				return nil, fmt.Errorf("read_csv: %q is not a number", f)
			}
			row = append(row, v)
		}
		if len(row) > 0 {
			rows = append(rows, row)
		}
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("read_csv: no numeric rows in %q", path)
	}
	cols := len(rows[0])
	flat := make([]float64, 0, len(rows)*cols)
	for i, row := range rows {
		if len(row) != cols {
			return nil, fmt.Errorf("read_csv: row %d has %d columns, expected %d", i+1, len(row), cols)
		}
		flat = append(flat, row...)
	}
	return tensor.New(flat, []int{len(rows), cols}), nil
}

func splitFields(line string) []string {
	return strings.FieldsFunc(line, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t' || r == '\r' || r == ';'
	})
}

// readFrame reads a header + numeric rows into a record of column tensors.
func (ip *Interp) readFrame(path string) (value.Value, error) {
	if !filepath.IsAbs(path) {
		path = filepath.Join(ip.currentDir(), path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read_frame: cannot read %q", path)
	}
	var header []string
	var rows [][]float64
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := splitFields(line)
		if header == nil {
			header = fields
			continue
		}
		if len(fields) != len(header) {
			return nil, fmt.Errorf("read_frame: row has %d values but %d columns", len(fields), len(header))
		}
		row := make([]float64, len(fields))
		for i, f := range fields {
			v, perr := strconv.ParseFloat(f, 64)
			if perr != nil {
				return nil, fmt.Errorf("read_frame: column %q value %q is not a number", header[i], f)
			}
			row[i] = v
		}
		rows = append(rows, row)
	}
	if header == nil {
		return nil, fmt.Errorf("read_frame: %q has no header row", path)
	}
	frame := value.NewRecord()
	for c, name := range header {
		col := make([]float64, len(rows))
		for r := range rows {
			col[r] = rows[r][c]
		}
		frame.Set(name, tensor.New(col, []int{len(rows)}))
	}
	return frame, nil
}

// writeFrame writes a record of equal-length 1-D column tensors to a CSV.
func (ip *Interp) writeFrame(frame *value.Record, path string) error {
	if len(frame.Keys) == 0 {
		return fmt.Errorf("write_frame: frame has no columns")
	}
	nrows := -1
	cols := make([][]float64, len(frame.Keys))
	for i, k := range frame.Keys {
		t, ok := value.AsTensor(frame.Fields[k])
		if !ok || len(t.Shape) != 1 {
			return fmt.Errorf("write_frame: column %q must be a 1-D tensor", k)
		}
		if nrows < 0 {
			nrows = len(t.Data)
		} else if len(t.Data) != nrows {
			return fmt.Errorf("write_frame: column %q has %d rows, expected %d", k, len(t.Data), nrows)
		}
		cols[i] = t.Data
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(ip.currentDir(), path)
	}
	var b strings.Builder
	b.WriteString(strings.Join(frame.Keys, ","))
	b.WriteByte('\n')
	for r := 0; r < nrows; r++ {
		for i := range cols {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(value.FormatNumber(cols[i][r]))
		}
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("write_frame: cannot write %q", path)
	}
	return nil
}

// mapLeaves applies f to each tensor leaf of tree, preserving structure.
func (ip *Interp) mapLeaves(f, tree value.Value) value.Value {
	switch t := tree.(type) {
	case value.Num, *tensor.Tensor:
		// A number is a numeric leaf like any other. Letting it fall to the
		// pass-through default would silently skip a scalar bias in an
		// optimiser's parameter tree.
		return ip.Apply(f, []value.Value{t}, 0)
	case *value.List:
		out := make([]value.Value, len(t.Items))
		for i, it := range t.Items {
			out[i] = ip.mapLeaves(f, it)
		}
		return &value.List{Items: out}
	case *value.Record:
		rec := value.NewRecord()
		for _, k := range t.Keys {
			rec.Set(k, ip.mapLeaves(f, t.Fields[k]))
		}
		return rec
	default:
		return tree // non-tensor leaves pass through unchanged
	}
}

// zipLeaves walks a list of same-shaped trees in parallel, calling f with the
// list of leaves at each position.
func (ip *Interp) zipLeaves(f value.Value, trees []value.Value) value.Value {
	if len(trees) == 0 {
		ip.panicf(0, "zip_leaves needs at least one tree")
	}
	switch first := trees[0].(type) {
	case value.Num, *tensor.Tensor:
		leaves := make([]value.Value, len(trees))
		for i, tr := range trees {
			if _, ok := value.AsTensor(tr); !ok {
				ip.panicf(0, "zip_leaves: trees have different structures")
			}
			leaves[i] = tr
		}
		return ip.Apply(f, []value.Value{&value.List{Items: leaves}}, 0)
	case *value.List:
		out := make([]value.Value, len(first.Items))
		for i := range first.Items {
			sub := make([]value.Value, len(trees))
			for k, tr := range trees {
				lst, ok := tr.(*value.List)
				if !ok || i >= len(lst.Items) {
					ip.panicf(0, "zip_leaves: trees have different structures")
				}
				sub[k] = lst.Items[i]
			}
			out[i] = ip.zipLeaves(f, sub)
		}
		return &value.List{Items: out}
	case *value.Record:
		rec := value.NewRecord()
		for _, key := range first.Keys {
			sub := make([]value.Value, len(trees))
			for k, tr := range trees {
				r, ok := tr.(*value.Record)
				if !ok {
					ip.panicf(0, "zip_leaves: trees have different structures")
				}
				v, present := r.Get(key)
				if !present {
					ip.panicf(0, "zip_leaves: record trees have different fields")
				}
				sub[k] = v
			}
			rec.Set(key, ip.zipLeaves(f, sub))
		}
		return rec
	default:
		return trees[0]
	}
}

// --- autodiff core ---------------------------------------------------------

type gradNode struct {
	leaf   *tensor.Tensor
	list   []*gradNode
	record *recordNode
	none   bool
}

type recordNode struct {
	keys  []string
	nodes map[string]*gradNode
}

func traceArg(v value.Value) (value.Value, *gradNode) {
	switch t := v.(type) {
	case value.Num:
		// grad(f)(3.0) passes a plain number, and without this it would fall to
		// the default below and report a gradient of zero for it.
		leaf := tensor.Leaf([]float64{float64(t)}, nil)
		return leaf, &gradNode{leaf: leaf}
	case *tensor.Tensor:
		leaf := tensor.Leaf(t.Data, t.Shape)
		return leaf, &gradNode{leaf: leaf}
	case *value.List:
		passed := make([]value.Value, len(t.Items))
		nodes := make([]*gradNode, len(t.Items))
		for i, it := range t.Items {
			passed[i], nodes[i] = traceArg(it)
		}
		return &value.List{Items: passed}, &gradNode{list: nodes}
	case *value.Record:
		rec := value.NewRecord()
		rn := &recordNode{keys: t.Keys, nodes: map[string]*gradNode{}}
		for _, k := range t.Keys {
			pv, node := traceArg(t.Fields[k])
			rec.Set(k, pv)
			rn.nodes[k] = node
		}
		return rec, &gradNode{record: rn}
	default:
		return v, &gradNode{none: true}
	}
}

func gradFromNode(n *gradNode) value.Value {
	if n.leaf != nil {
		g := make([]float64, len(n.leaf.Data))
		if n.leaf.Grad != nil {
			copy(g, n.leaf.Grad)
		}
		shape := make([]int, len(n.leaf.Shape))
		copy(shape, n.leaf.Shape)
		return tensor.New(g, shape)
	}
	if n.list != nil {
		items := make([]value.Value, len(n.list))
		for i, c := range n.list {
			items[i] = gradFromNode(c)
		}
		return &value.List{Items: items}
	}
	if n.record != nil {
		rec := value.NewRecord()
		for _, k := range n.record.keys {
			rec.Set(k, gradFromNode(n.record.nodes[k]))
		}
		return rec
	}
	return tensor.Scalar(0)
}

// gradients runs f with gradient-tracking arguments and returns the scalar
// value, the gradient of the first argument, and the gradients of all args.
// refuseNestedGrad rejects grad(grad(f)) instead of answering it wrongly.
//
// The gradient grad returns is a value, not a traced expression: reverse mode
// reads the numbers out of the graph and hands them back with no history
// attached. Differentiating that again differentiates a constant, so the answer
// is zero, and it is zero silently, which is the worst way for a derivative to
// be wrong. Somebody building on a second derivative would have no reason to
// look.
//
// Second derivatives are hessian's job, which runs forward mode over the reverse
// pass and gets it right. This points there rather than quietly failing.
func refuseNestedGrad(caller string, f value.Value) error {
	b, ok := f.(*value.Builtin)
	if !ok {
		return nil
	}
	if b.Name != "grad(fn)" && b.Name != "grads(fn)" {
		return nil
	}
	return fmt.Errorf(
		"%s(%s) is not a second derivative: the gradient %s returns is a plain value with no "+
			"history, so differentiating it again gives zero. Use hessian(f) for a second "+
			"derivative, or jacobian(f) for the matrix of first ones",
		caller, b.Name, b.Name)
}

func (ip *Interp) gradients(f value.Value, callArgs []value.Value) (*tensor.Tensor, value.Value, []value.Value, error) {
	if len(callArgs) == 0 {
		return nil, nil, nil, fmt.Errorf("gradient function requires at least one argument")
	}
	passArgs := make([]value.Value, len(callArgs))
	nodes := make([]*gradNode, len(callArgs))
	for i, v := range callArgs {
		passArgs[i], nodes[i] = traceArg(v)
	}

	out := ip.Apply(f, passArgs, 0)
	ot, ok := value.AsTensor(out)
	if !ok || !ot.IsScalar() {
		return nil, nil, nil, fmt.Errorf("grad target must return a scalar")
	}
	if err := ot.Backward(); err != nil {
		return nil, nil, nil, err
	}

	all := make([]value.Value, len(nodes))
	for i, n := range nodes {
		all[i] = gradFromNode(n)
	}
	return tensor.Scalar(ot.Data[0]), gradFromNode(nodes[0]), all, nil
}

// jacobian computes the Jacobian of f at x by running one reverse-mode pass per
// output component: row j is the gradient of the j-th output w.r.t. x. The
// result is an [m, n] tensor for an m-vector output and an n-element input.
func (ip *Interp) jacobian(f value.Value, x value.Value) (value.Value, error) {
	xt, ok := value.AsTensor(x)
	if !ok {
		return nil, fmt.Errorf("jacobian: the input must be a tensor")
	}
	n := len(xt.Data)
	// A gradient-free pass to learn the output size.
	y0, err := ip.applyToTensor(f, tensor.New(xt.Data, xt.Shape))
	if err != nil {
		return nil, err
	}
	m := len(y0.Data)
	jac := make([]float64, m*n)
	for j := 0; j < m; j++ {
		leaf := tensor.Leaf(xt.Data, xt.Shape)
		y, err := ip.applyToTensor(f, leaf)
		if err != nil {
			return nil, err
		}
		if len(y.Data) != m {
			return nil, fmt.Errorf("jacobian: f returned a different output size on re-evaluation")
		}
		// Isolate the j-th output as a scalar and backpropagate it.
		sel := make([]float64, m)
		sel[j] = 1
		picked, err := tensor.Mul(y, tensor.New(sel, append([]int{}, y.Shape...)))
		if err != nil {
			return nil, err
		}
		s := tensor.Sum(picked)
		if err := s.Backward(); err != nil {
			return nil, err
		}
		if leaf.Grad != nil {
			copy(jac[j*n:(j+1)*n], leaf.Grad)
		}
	}
	return tensor.New(jac, []int{m, n}), nil
}

// hessian computes the Hessian of a scalar function f at x by forward-mode
// second-order autodiff over the graph built from a single input leaf.
func (ip *Interp) hessian(f value.Value, x value.Value) (value.Value, error) {
	xt, ok := value.AsTensor(x)
	if !ok {
		return nil, fmt.Errorf("hessian: the input must be a tensor")
	}
	leaf := tensor.Leaf(xt.Data, xt.Shape)
	// Record forward-mode (jet) closures only while building this graph, so
	// ordinary training and grad pay nothing for second-order support.
	tensor.SetRecordJets(true)
	defer tensor.SetRecordJets(false)
	out, err := ip.applyToTensor(f, leaf)
	if err != nil {
		return nil, err
	}
	h, n, err := tensor.Hessian(out, leaf)
	if err != nil {
		return nil, err
	}
	return tensor.New(h, []int{n, n}), nil
}

// applyToTensor calls a one-argument function and requires a tensor result.
func (ip *Interp) applyToTensor(f value.Value, arg *tensor.Tensor) (*tensor.Tensor, error) {
	out := ip.Apply(f, []value.Value{arg}, 0)
	yt, ok := value.AsTensor(out)
	if !ok {
		return nil, fmt.Errorf("jacobian: f must return a tensor, got %s", value.Format(out))
	}
	return yt, nil
}

// --- argument coercion -----------------------------------------------------

func asTensor(v value.Value, who string) (*tensor.Tensor, error) {
	if t, ok := value.AsTensor(v); ok {
		return t, nil
	}
	return nil, fmt.Errorf("%s expects a tensor/number", who)
}

// scalarOf reads a single number. It reads a Num straight out rather than
// widening it first, because a builtin that wants a scalar wants an axis or a
// bound and would throw the tensor away again.
func scalarOf(v value.Value, who string) (float64, error) {
	if n, ok := value.AsNumber(v); ok {
		return n, nil
	}
	if _, numeric := value.AsTensor(v); numeric {
		return 0, fmt.Errorf("%s expects a scalar", who)
	}
	return 0, fmt.Errorf("%s expects a tensor/number", who)
}

// isScalarValue reports whether v is a single number, as opposed to a sequence
// of them. Used where a scalar and a one-element sequence have to mean
// different things.
func isScalarValue(v value.Value) bool {
	if _, ok := v.(value.Num); ok {
		return true
	}
	t, ok := v.(*tensor.Tensor)
	return ok && t.Size() == 1
}

func intOf(v value.Value, who string) (int, error) {
	f, err := scalarOf(v, who)
	if err != nil {
		return 0, err
	}
	return int(math.Trunc(f)), nil
}

// indicesOf reads a sequence of integer indices from either a list (of scalars)
// or a 1-D tensor.
func indicesOf(v value.Value, who string) ([]int, error) {
	switch t := v.(type) {
	case value.Num:
		return []int{int(math.Trunc(float64(t)))}, nil
	case *value.List:
		out := make([]int, len(t.Items))
		for i, it := range t.Items {
			n, err := intOf(it, who)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case *tensor.Tensor:
		if len(t.Shape) > 1 {
			return nil, fmt.Errorf("%s expects a 1-D tensor or list of indices, got shape %v", who, t.Shape)
		}
		out := make([]int, len(t.Data))
		for i, f := range t.Data {
			out[i] = int(math.Trunc(f))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%s expects a list or 1-D tensor of indices", who)
	}
}

func shapeFromArgs(args []value.Value, who string) ([]int, error) {
	if len(args) == 1 {
		if lst, ok := args[0].(*value.List); ok {
			dims := make([]int, len(lst.Items))
			for i, it := range lst.Items {
				d, err := intOf(it, who)
				if err != nil {
					return nil, err
				}
				dims[i] = d
			}
			return dims, nil
		}
	}
	dims := make([]int, len(args))
	for i, a := range args {
		d, err := intOf(a, who)
		if err != nil {
			return nil, err
		}
		dims[i] = d
	}
	return dims, nil
}

func toItems(v value.Value, who string) ([]value.Value, error) {
	switch t := v.(type) {
	case *value.List:
		return t.Items, nil
	case *tensor.Tensor:
		if len(t.Shape) == 1 {
			out := make([]value.Value, len(t.Data))
			for i, x := range t.Data {
				out[i] = tensor.Scalar(x)
			}
			return out, nil
		}
	}
	return nil, fmt.Errorf("%s expects a list or 1-D tensor", who)
}

func valueToNested(v value.Value) (any, error) {
	switch t := v.(type) {
	case value.Num:
		return float64(t), nil
	case *tensor.Tensor:
		return t.ToNested(), nil
	case *value.List:
		out := make([]any, len(t.Items))
		for i, it := range t.Items {
			n, err := valueToNested(it)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot convert value to a tensor")
}

func randomTensor(shape []int, sample func() float64) *tensor.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float64, n)
	for i := range data {
		data[i] = sample()
	}
	return tensor.New(data, shape)
}
