package interp_test

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/martin-k-m/twill/internal/gbm"
	"github.com/martin-k-m/twill/internal/interp"
	"github.com/martin-k-m/twill/internal/tensor"
	"github.com/martin-k-m/twill/internal/value"
)

// roundTrip saves v to a temp file via the language's save/load builtins and
// returns the loaded value.
func roundTrip(t *testing.T, v value.Value) value.Value {
	t.Helper()
	path := filepath.Join(t.TempDir(), "v.bin")
	ip := interp.New(func(string) {})
	ip.Global.Define("x", v)
	ip.Global.Define("p", value.Str(path))
	if _, err := ip.Run(`save(x, p)`); err != nil {
		t.Fatalf("save: %v", err)
	}
	out, err := ip.Run(`load(p)`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return out
}

func tensorsEqual(a, b *tensor.Tensor) bool {
	if len(a.Shape) != len(b.Shape) || len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

func TestSaveLoadTensor(t *testing.T) {
	orig := tensor.New([]float64{1.5, -2, 3.25, 0, math.Pi, 1e-9}, []int{2, 3})
	got, ok := roundTrip(t, orig).(*tensor.Tensor)
	if !ok || !tensorsEqual(orig, got) {
		t.Fatalf("tensor round-trip mismatch: %v", got)
	}
}

func TestSaveLoadScalarAndScalars(t *testing.T) {
	orig := tensor.Scalar(42.5)
	got, ok := roundTrip(t, orig).(*tensor.Tensor)
	if !ok || !got.IsScalar() || got.Data[0] != 42.5 {
		t.Fatalf("scalar round-trip mismatch: %v", got)
	}
}

func TestSaveLoadRecordOfTensors(t *testing.T) {
	rec := value.NewRecord()
	rec.Set("w", tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2}))
	rec.Set("b", tensor.New([]float64{0.1, 0.2, 0.3}, []int{3}))
	rec.Set("name", value.Str("layer0"))
	got, ok := roundTrip(t, rec).(*value.Record)
	if !ok {
		t.Fatalf("expected a record, got %T", got)
	}
	if len(got.Keys) != 3 || got.Keys[0] != "w" || got.Keys[2] != "name" {
		t.Fatalf("record keys not preserved in order: %v", got.Keys)
	}
	w, _ := got.Get("w")
	if !tensorsEqual(w.(*tensor.Tensor), rec.Fields["w"].(*tensor.Tensor)) {
		t.Fatal("record field w mismatch")
	}
	if s, _ := got.Get("name"); s != value.Str("layer0") {
		t.Fatalf("record string field mismatch: %v", s)
	}
}

func TestSaveLoadNestedListRecord(t *testing.T) {
	// A pytree like a model held as a list of [weight, bias] layers.
	layer := &value.List{Items: []value.Value{
		tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}),
		tensor.New([]float64{5, 6}, []int{2}),
	}}
	model := &value.List{Items: []value.Value{layer, value.Bool(true)}}
	got, ok := roundTrip(t, model).(*value.List)
	if !ok || len(got.Items) != 2 {
		t.Fatalf("list round-trip shape wrong: %v", got)
	}
	inner := got.Items[0].(*value.List)
	if !tensorsEqual(inner.Items[1].(*tensor.Tensor), tensor.New([]float64{5, 6}, []int{2})) {
		t.Fatal("nested tensor mismatch")
	}
	if got.Items[1] != value.Bool(true) {
		t.Fatal("nested bool mismatch")
	}
}

func TestSaveLoadGBMModel(t *testing.T) {
	// Train a small model, round-trip it, and require identical predictions.
	n, d := 200, 4
	X := make([]float64, n*d)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		var s float64
		for j := 0; j < d; j++ {
			// deterministic pseudo-features (no rng needed)
			v := math.Sin(float64(i*7+j*13)) * 0.5
			X[i*d+j] = v
			s += float64(j+1) * v
		}
		if s > 0 {
			y[i] = 1
		}
	}
	p := gbm.DefaultParams()
	p.Rounds = 30
	p.Objective = gbm.Logistic
	m, err := gbm.Fit(X, y, n, d, p)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := roundTrip(t, m).(*gbm.Model)
	if !ok {
		t.Fatalf("expected a model, got %T", got)
	}
	p1, _ := m.Predict(X, n, d)
	p2, _ := got.Predict(X, n, d)
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Fatalf("prediction %d differs after round-trip: %v != %v", i, p1[i], p2[i])
		}
	}
}
