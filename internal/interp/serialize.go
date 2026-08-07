package interp

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/martin-k-m/twill/internal/gbm"
	"github.com/martin-k-m/twill/internal/tensor"
	"github.com/martin-k-m/twill/internal/value"
)

// A saved file is: the 4-byte magic "RSTR", a version byte, then one tagged
// value. The format is little-endian and exact (float64 bit patterns), so a
// round-trip reproduces a value's numbers bit-for-bit.
//
// The magic stays "RSTR" even though the language was renamed from Raster to
// Twill. These four bytes are a compatibility contract, not branding: every
// file a user has already saved must keep loading, and nothing in the language,
// the CLI or the docs ever exposes them. A format named after a former name is
// ordinary (PK in zip, the chunk names in PNG, ELF). Accepting two magics would
// make the reader strictly worse for no gain. If the format ever changes for a
// real reason, that is the moment to introduce magic "TWIL" at version 2 and
// write one dual-path reader, rather than paying for two.
var saveMagic = [4]byte{'R', 'S', 'T', 'R'}

const saveVersion byte = 1

// Value tags.
const (
	tagTensor byte = 'T'
	tagList   byte = 'L'
	tagRecord byte = 'R'
	tagBool   byte = 'B'
	tagStr    byte = 'S'
	tagUnit   byte = 'U'
	tagModel  byte = 'G' // a gbm.Model
)

func saveValue(v value.Value, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	if _, err := w.Write(saveMagic[:]); err != nil {
		return err
	}
	if err := w.WriteByte(saveVersion); err != nil {
		return err
	}
	if err := writeValue(w, v); err != nil {
		return err
	}
	return w.Flush()
}

func loadValue(path string) (value.Value, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var m [4]byte
	if _, err := io.ReadFull(r, m[:]); err != nil {
		return nil, err
	}
	if m != saveMagic {
		return nil, fmt.Errorf("not a Twill save file")
	}
	ver, err := r.ReadByte()
	if err != nil {
		return nil, err
	}
	if ver != saveVersion {
		return nil, fmt.Errorf("unsupported save-file version %d (this build writes %d)", ver, saveVersion)
	}
	return readValue(r)
}

func writeValue(w *bufio.Writer, v value.Value) error {
	switch t := v.(type) {
	case value.Num:
		// Saved as the rank-0 tensor it stands for, so the file format did not
		// change and an old file still loads.
		return writeValue(w, tensor.Scalar(float64(t)))
	case *tensor.Tensor:
		if err := w.WriteByte(tagTensor); err != nil {
			return err
		}
		dims := make([]int32, len(t.Shape))
		for i, d := range t.Shape {
			dims[i] = int32(d)
		}
		return writeChunks(w, int32(len(dims)), dims, int64(len(t.Data)), t.Data)
	case *value.List:
		if err := w.WriteByte(tagList); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, int32(len(t.Items))); err != nil {
			return err
		}
		for _, it := range t.Items {
			if err := writeValue(w, it); err != nil {
				return err
			}
		}
		return nil
	case *value.Record:
		if err := w.WriteByte(tagRecord); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, int32(len(t.Keys))); err != nil {
			return err
		}
		for _, k := range t.Keys {
			if err := writeString(w, k); err != nil {
				return err
			}
			if err := writeValue(w, t.Fields[k]); err != nil {
				return err
			}
		}
		return nil
	case value.Bool:
		b := byte(0)
		if t {
			b = 1
		}
		_, err := w.Write([]byte{tagBool, b})
		return err
	case value.Str:
		if err := w.WriteByte(tagStr); err != nil {
			return err
		}
		return writeString(w, string(t))
	case value.Unit:
		return w.WriteByte(tagUnit)
	case *gbm.Model:
		if err := w.WriteByte(tagModel); err != nil {
			return err
		}
		data, err := t.MarshalBinary()
		if err != nil {
			return err
		}
		return writeChunks(w, int32(len(data)), data)
	default:
		return fmt.Errorf("cannot save a value of this kind (%s)", value.Format(v))
	}
}

func readValue(r *bufio.Reader) (value.Value, error) {
	tag, err := r.ReadByte()
	if err != nil {
		return nil, err
	}
	switch tag {
	case tagTensor:
		var rank int32
		if err := binary.Read(r, binary.LittleEndian, &rank); err != nil {
			return nil, err
		}
		if rank < 0 || rank > 16 {
			return nil, fmt.Errorf("corrupt save file (tensor rank %d)", rank)
		}
		dims := make([]int32, rank)
		if err := binary.Read(r, binary.LittleEndian, dims); err != nil {
			return nil, err
		}
		var n int64
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		if n < 0 || n > (1<<40) {
			return nil, fmt.Errorf("corrupt save file (tensor size %d)", n)
		}
		data := make([]float64, n)
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		shape := make([]int, rank)
		for i, d := range dims {
			shape[i] = int(d)
		}
		return tensor.New(data, shape), nil
	case tagList:
		n, err := readCount(r)
		if err != nil {
			return nil, err
		}
		items := make([]value.Value, n)
		for i := range items {
			v, err := readValue(r)
			if err != nil {
				return nil, err
			}
			items[i] = v
		}
		return &value.List{Items: items}, nil
	case tagRecord:
		n, err := readCount(r)
		if err != nil {
			return nil, err
		}
		rec := value.NewRecord()
		for i := 0; i < n; i++ {
			k, err := readString(r)
			if err != nil {
				return nil, err
			}
			v, err := readValue(r)
			if err != nil {
				return nil, err
			}
			rec.Set(k, v)
		}
		return rec, nil
	case tagBool:
		b, err := r.ReadByte()
		if err != nil {
			return nil, err
		}
		return value.Bool(b != 0), nil
	case tagStr:
		s, err := readString(r)
		if err != nil {
			return nil, err
		}
		return value.Str(s), nil
	case tagUnit:
		return value.TheUnit, nil
	case tagModel:
		var n int32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		if n < 0 {
			return nil, fmt.Errorf("corrupt save file (model size %d)", n)
		}
		data := make([]byte, n)
		if _, err := io.ReadFull(r, data); err != nil {
			return nil, err
		}
		m := &gbm.Model{}
		if err := m.UnmarshalBinary(data); err != nil {
			return nil, err
		}
		return m, nil
	default:
		return nil, fmt.Errorf("corrupt save file (unknown value tag %q)", tag)
	}
}

// writeChunks writes each value in order (little-endian). Used for the
// fixed-layout parts of a tensor or model header.
func writeChunks(w *bufio.Writer, vs ...any) error {
	for _, v := range vs {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

func writeString(w *bufio.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, int32(len(s))); err != nil {
		return err
	}
	_, err := w.WriteString(s)
	return err
}

func readString(r *bufio.Reader) (string, error) {
	n, err := readCount(r)
	if err != nil {
		return "", err
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readCount reads a non-negative int32 length, rejecting corrupt values.
func readCount(r *bufio.Reader) (int, error) {
	var n int32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return 0, err
	}
	if n < 0 {
		return 0, fmt.Errorf("corrupt save file (negative length %d)", n)
	}
	return int(n), nil
}
