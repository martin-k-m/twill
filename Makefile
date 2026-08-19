BINARY := twill
PKG := ./cmd/twill

.PHONY: build test vet fmt race lint check ci bench examples install clean

build:
	go build -o $(BINARY) $(PKG)

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

# The race pass, with the same flags and budget CI uses. -short skips the two
# model-training examples and the self-hosted differential runs, which is what
# keeps this inside the timeout; `test` above runs all of them.
race:
	go test -race -short -timeout 25m ./internal/tensor/ ./internal/interp/

# The two linters CI runs. They fetch a tool, so they need the network, which is
# why they are not in `check`.
lint:
	out="$$(go run golang.org/x/tools/cmd/deadcode@latest -test ./...)"; if [ -n "$$out" ]; then echo "$$out"; echo "dead code found"; exit 1; fi
	go run honnef.co/go/tools/cmd/staticcheck@latest ./...

# What CI runs, minus the two linters that need the network. Run this before
# tagging a release.
#
# The race pass is in here because leaving it out is how 1.6.5 shipped with a CI
# failure: the local gate was `vet test` and gofmt, CI was that plus a race pass
# and two linters, and the suite had grown past the race budget without anything
# local saying so. A gate that does not match CI is a gate that lets things
# through.
check: build vet test race
	gofmt -l . | tee /dev/stderr | (! read)

# Everything, including the linters. What the release gate should be.
ci: check lint

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
