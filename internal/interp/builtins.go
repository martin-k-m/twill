package interp

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/martin-k-m/raster/internal/gbm"
	"github.com/martin-k-m/raster/internal/tensor"
	"github.com/martin-k-m/raster/internal/value"
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

	def("argmax", -1, true, func(a []value.Value) (value.Value, error) {
		t, err := asTensor(a[0], "argmax")
		if err != nil {
			return nil, err
		}
		axis := len(t.Shape) - 1
		if len(a) == 2 {
			axis, err = intOf(a[1], "argmax")
			if err != nil {
				return nil, err
			}
		}
		return tensor.ArgmaxAxis(t, axis)
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

	// Automatic differentiation.
	def("grad", 1, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
		return &value.Builtin{Name: "grad(fn)", Variadic: true, Arity: -1, Fn: func(call []value.Value) (value.Value, error) {
			_, grad0, _, err := ip.gradients(f, call)
			return grad0, err
		}}, nil
	})

	def("grads", 1, false, func(a []value.Value) (value.Value, error) {
		f := a[0]
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
		}
		return nil, fmt.Errorf("len expects a tensor or list")
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
	// curves, and running peaks for backtests. Forward-only (no gradient).
	cumOp := func(name string, f func(prev, x float64) float64) {
		def(name, 1, false, func(a []value.Value) (value.Value, error) {
			t, err := asTensor(a[0], name)
			if err != nil {
				return nil, err
			}
			out := make([]float64, len(t.Data))
			for i, v := range t.Data {
				if i == 0 {
					out[i] = v
				} else {
					out[i] = f(out[i-1], v)
				}
			}
			return tensor.New(out, append([]int{}, t.Shape...)), nil
		})
	}
	cumOp("cumsum", func(prev, x float64) float64 { return prev + x })
	cumOp("cumprod", func(prev, x float64) float64 { return prev * x })
	cumOp("cummax", math.Max)
	cumOp("cummin", math.Min)

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
		t, ok := frame.Fields[k].(*tensor.Tensor)
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
	case *tensor.Tensor:
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
	case *tensor.Tensor:
		leaves := make([]value.Value, len(trees))
		for i, tr := range trees {
			if _, ok := tr.(*tensor.Tensor); !ok {
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
	ot, ok := out.(*tensor.Tensor)
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

// --- argument coercion -----------------------------------------------------

func asTensor(v value.Value, who string) (*tensor.Tensor, error) {
	if t, ok := v.(*tensor.Tensor); ok {
		return t, nil
	}
	return nil, fmt.Errorf("%s expects a tensor/number", who)
}

func scalarOf(v value.Value, who string) (float64, error) {
	t, err := asTensor(v, who)
	if err != nil {
		return 0, err
	}
	if t.Size() != 1 {
		return 0, fmt.Errorf("%s expects a scalar", who)
	}
	return t.Data[0], nil
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
