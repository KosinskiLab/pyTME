Execute get_rotational_sets.sh to acquire rotational sets from https://github.com/cffk/orientation/releases/tag/v1.1

Rotational sets in this folder c*.npy are under the following license.

The MIT License (MIT)

Copyright (c) 2006-2015, Charles Karney, SRI International

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Taken from https://github.com/cffk/orientation/ README.md

The files are in the following format:

Any number of initial comment lines beginning with “#”.
A line containing either “format quaternion” or “format euler”.
A line containing: N α(°) c.
N lines containing: q0i q1i q2i q3i wi (for quaternions)
or N lines containing: ai bi gi wi (for Euler angles).

Suboptimal sets were removed as indicated in the README

The following orientation sets are non-optimal:

c48u9 (beaten by c48n9),
c600vec (beaten by c48u27),
c48u309 (beaten by c48n309),
c48u527 (beaten by c48n527).
The following orientation sets are sub-optimal with a substantially thinner covering achieved by another set with somewhat more points:

c48u157 (use c48u181 instead),
c48u519 (use c48n527 instead),
c48u2867 (use c48u2947 instead),
c48u4701 (use c48u4749 instead).

A quaternion [<i>q</i><sub>0</sub>, <i>q</i><sub>1</sub>,
<i>q</i><sub>2</sub>, <i>q</i><sub>3</sub>] represents the rotation give
by the matrix whose components are
Taken from: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
<blockquote>
<table>
<tr align=center>
<td> 1 &minus; 2<i>q</i><sub>2</sub><sup>2</sup> + 2<i>q</i><sub>3</sub><sup>2</sup>
<td> 2<i>q</i><sub>1</sub><i>q</i><sub>2</sub> &minus; 2<i>q</i><sub>0</sub><i>q</i><sub>3</sub>
<td> 2<i>q</i><sub>1</sub><i>q</i><sub>3</sub> + 2<i>q</i><sub>0</sub><i>q</i><sub>2</sub>
<tr align=center>
<td> 2<i>q</i><sub>2</sub><i>q</i><sub>1</sub> + 2<i>q</i><sub>0</sub><i>q</i><sub>3</sub>
<td> 1 &minus; 2<i>q</i><sub>3</sub><sup>2</sup> + 2<i>q</i><sub>1</sub><sup>2</sup>
<td> 2<i>q</i><sub>2</sub><i>q</i><sub>3</sub> &minus; 2<i>q</i><sub>0</sub><i>q</i><sub>1</sub>
<tr align=center>
<td> 2<i>q</i><sub>3</sub><i>q</i><sub>1</sub> &minus; 2<i>q</i><sub>0</sub><i>q</i><sub>2</sub>
<td> 2<i>q</i><sub>3</sub><i>q</i><sub>2</sub> + 2<i>q</i><sub>0</sub><i>q</i><sub>1</sub>
<td> 1 &minus; 2<i>q</i><sub>1</sub><sup>2</sup> + 2<i>q</i><sub>2</sub><sup>2</sup>
</table>
</blockquote>