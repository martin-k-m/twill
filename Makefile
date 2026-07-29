BINARY := raster
PKG := ./cmd/raster

.PHONY: build test vet fmt check bench examples install clean

build:
	go build -o $(BINARY) $(PKG)

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

# What CI runs.
check: vet test
	gofmt -l . | tee /dev/stderr | (! read)

bench:
	go test -run=XXX -bench=. ./internal/tensor/

examples: build
	./$(BINARY) examples/hello.ra
	./$(BINARY) examples/autodiff.ra
	./$(BINARY) examples/linreg.ra
	./$(BINARY) examples/nn_xor.ra
	./$(BINARY) examples/classifier.ra
	./$(BINARY) check examples/shapes.ra

install:
	go install $(PKG)

clean:
	rm -f $(BINARY) $(BINARY).exe
