BINARY := twill
PKG := ./cmd/twill

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
	./$(BINARY) examples/hello.tw
	./$(BINARY) examples/autodiff.tw
	./$(BINARY) examples/linreg.tw
	./$(BINARY) examples/nn_xor.tw
	./$(BINARY) examples/classifier.tw
	./$(BINARY) check examples/shapes.tw

install:
	go install $(PKG)

clean:
	rm -f $(BINARY) $(BINARY).exe
