package interp

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/martin-k-m/aster/internal/tensor"
	"github.com/martin-k-m/aster/internal/value"
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
	unaryOp("sum", tensor.Sum)
	unaryOp("mean", tensor.Mean)

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
		return randomTensor(shape, func() float64 { return rand.NormFloat64() }), nil
	})
	def("rand", -1, true, func(a []value.Value) (value.Value, error) {
		shape, err := shapeFromArgs(a, "rand")
		if err != nil {
			return nil, err
		}
		return randomTensor(shape, rand.Float64), nil
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
	def("transpose", 1, false, func(a []value.Value) (value.Value, error) {
		t, err := asTensor(a[0], "transpose")
		if err != nil {
			return nil, err
		}
		return tensor.Transpose(t)
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
}

// --- autodiff core ---------------------------------------------------------

type gradNode struct {
	leaf *tensor.Tensor
	list []*gradNode
	none bool
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
