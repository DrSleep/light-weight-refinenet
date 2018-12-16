"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division

cimport cython
cimport numpy as np
import numpy as np

def fast_cm(unsigned char[::1] preds, unsigned char[::1] gt,
            int n_classes):
    """Computing confusion matrix faster.

    Args:
      preds (Tensor) : predictions (either flatten or of size (len(gt), top-N)).
      gt (Tensor) : flatten gt.
      n_classes (int) : number of classes.

    Returns:

      Confusion matrix
      (Tensor of size (n_classes, n_classes)).

    """
    cdef np.ndarray[np.int_t, ndim=2] cm = np.zeros((n_classes, n_classes),
                                                    dtype=np.int_)
    cdef np.intp_t i,a,p, n = gt.shape[0]

    for i in range(n):
        a = gt[i]
        p = preds[i]
        cm[a, p] += 1
    return cm

def compute_iu(np.ndarray[np.int_t, ndim=2] cm):
    """Compute IU from confusion matrix.

    Args:
      cm (Tensor) : square confusion matrix.

    Returns:
      IU vector (Tensor).

    """
    cdef unsigned int pi = 0
    cdef unsigned int gi = 0
    cdef unsigned int ii = 0
    cdef unsigned int denom = 0
    cdef unsigned int n_classes = cm.shape[0]
    cdef np.ndarray[np.float_t, ndim=1] IU = np.ones(n_classes)
    cdef np.intp_t i
    for i in xrange(n_classes):
        pi = sum(cm[:, i])
        gi = sum(cm[i, :])
        ii = cm[i, i]
        denom = pi + gi - ii
        if denom > 0:
            IU[i] = ii / denom
    return IU
