package dimred

import (
	"fmt"
	"log"
	"math"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"github.com/brookluers/dstream/dstream"
)

type LDA struct {
	*MomentStream

	// Name of the binary response variable
	responseVarName string

	// Position of the binary response variable
	responseVarPos int

	// Positions of all other variables
	xpos []int

	// The min(k-1, p) discriminant directions
	ldirs [][]float64
	
	// The within-class covariance estimate
	wmat [][]float64

	// The proportion of samples in each class
	pihat []float64
	
	// The number of classes
	k int

	projDim int
}

// LDir returns an estimated discriminant direction
// for use in LDA classification or dimension reduction for optimal class separation
func (lda *LDA) LDir(j int) []float64 {

	return lda.ldirs[j]

}

// NewLDA returns a new LDA value for the given data stream.
func NewLDA(data dstream.Dstream, response string) *LDA {

	d := &LDA{
		MomentStream: NewMomentStream(data, response),
	}

	return d
}

func (lda *LDA) Done() *LDA {
	lda.MomentStream.Done()
	lda.calcMargMean()
	lda.calcMargCov()
	return lda
}

// calcWithinCov computes 
func (lda *LDA) calcWithinCov() *LDA {
     //the within-class covariance matrix
     return lda
}

func (lda *LDA) calcBetweenCov() *LDA {
     // the between-class covariance matrix
     return lda
}

func (lda *LDA) Fit() {

	p := lda.Dim()
	pp := p * p

	margcov := mat.NewSymDense(p, lda.GetMargCov())
	msr := new(mat.Cholesky)
	if ok := msr.Factorize(margcov); !ok {
		print("Can't factorize marginal covariance")
		panic("")
	}

	if lda.log != nil {
	   // logging....
	}
	
	lda.ldirs = make([][]float64,  // min(k-1, p))

	if lda.log != nil {

	}
}

func (lda *LDA) SetLogFile(filename string) *LDA {
	fid, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	lda.log = log.New(fid, "", log.Lshortfile)

	return lda
}
