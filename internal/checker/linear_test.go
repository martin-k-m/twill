package checker_test

import "testing"

// linear(x, W) = x @ Wᵀ: valid shapes type-check, a bad inner dim is caught.
func TestLinearShapeChecks(t *testing.T) {
	// x [2,3] against W [4,3] -> [2,4], fine.
	wantNone(t, "let W = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0],[1.0,0.0,1.0]]\nlet x = [[1.0,2.0,3.0],[4.0,5.0,6.0]]\nlet y = linear(x, W)")
	// 1-D input [3] against W [4,3] -> [4], fine.
	wantNone(t, "let W = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0],[1.0,0.0,1.0]]\nlet x = [1.0,2.0,3.0]\nlet y = linear(x, W)")
	// inner dims disagree: x last dim 3, W last dim 2.
	wantOne(t, "let W = [[1.0,2.0]]\nlet x = [1.0,2.0,3.0]\nlet y = linear(x, W)", "inner")
}
