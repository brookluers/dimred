package dimred

import (
	"fmt"
	"log"
	//"math"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"github.com/brookluers/dstream/dstream"
)

type LDA struct {
	*MomentStream

	// The min(k-1, p) discriminant directions
	ldirs [][]float64
	
	// The within-class covariance estimate
	wmat []float64

	// The between-class covariance estimate (covariance of the means)
	bmat []float64

	// The proportion of samples in each class
	pihat []float64
	
	// The number of classes
	k int

}

// LDir returns an estimated discriminant direction
// for use in LDA classification or dimension reduction for optimal class separation
func (lda *LDA) LDir(j int) []float64 {
	return lda.ldirs[j]
}

// NewLDA returns a new LDA value for the given data stream.
func NewLDA(data dstream.Dstream, response string, nclass int) *LDA {

	d := &LDA {
		MomentStream: NewMomentStream(data, response),
		k: nclass,
	}
	return d
}

func (lda *LDA) Done() *LDA {
	lda.MomentStream.Done()
	lda.calcMargMean()
	fmt.Printf("margmean = %v\n", lda.MomentStream.GetMargMean())
	fmt.Printf("GetMean(0) = %v\n", lda.MomentStream.GetMean(0))
	lda.calcMargCov()    
	fmt.Printf("Nobs = %v\n", lda.Data.NumObs())
	lda.calcPi()
	lda.calcWithinCov()
	lda.calcBetweenCov()
	return lda
}

func (lda *LDA) calcPi() *LDA {
     pi := make([]float64, lda.k)
     for j := 0; j < lda.k; j++{
     	 pi[j] = float64(lda.GetN(j)) / float64(lda.Data.NumObs())
     }
     lda.pihat = pi

     return lda
}

// calcWithinCov computes the within-class covariance matrix
func (lda *LDA) calcWithinCov() *LDA {
     fmt.Printf("called calcWithinCov()\n")
     p := lda.Dim()
     pp := p * p
     N := lda.Data.NumObs()
     w := make([]float64, pp)

     for j := 0; j < lda.k; j++{
	 floats.AddScaled(w, float64(lda.GetN(j)), lda.GetCov(j))
     }
     floats.Scale(1/float64(N - lda.k), w)

     lda.wmat = w

     return lda
}

func (lda *LDA) GetWCov() []float64 {
     return lda.wmat
}

func (lda *LDA) GetBCov() []float64 {
     return lda.bmat
}

func (lda *LDA) calcBetweenCov() *LDA {

     // the between-class covariance matrix
     p := lda.Dim()
     pp := p * p
     N := lda.Data.NumObs()
     bb :=  make([]float64, pp)

     for j := 0; j < lda.k; j++{
     	 classmean := lda.GetMean(j)
	 Nk := lda.GetN(j)
     	 for k1 := 0; k1 < p; k1++ {
	     u1 := classmean[k1] - lda.GetMargMean()[k1]
	     for k2 := 0; k2 <= k1; k2++ {
	     	 u2 := classmean[k2] - lda.GetMargMean()[k2]
		 bb[k1*p+k2] += float64(Nk) * u1 * u2
	     }
	 }
     }
     reflect(bb, p)
     floats.Scale(1/float64(N), bb)
     lda.bmat = bb

     return lda
}

func (lda *LDA) Fit() {

	p := lda.Dim()
	// pp := p * p

	margcov := mat.NewSymDense(p, lda.GetMargCov())
	msr := new(mat.Cholesky)
	if ok := msr.Factorize(margcov); !ok {
		print("Can't factorize marginal covariance")
		panic("")
	}

	// lda.ldirs = make([][]float64,min(k-1, p))

	if lda.log != nil {
	   lda.log.Printf("Writing to the log file\n")
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
