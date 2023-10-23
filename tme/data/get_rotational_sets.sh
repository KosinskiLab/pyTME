#!/bin/bash
VERSION=1.1

curl -L \
	https://github.com/cffk/orientation/archive/refs/tags/v${VERSION}.zip \
	--output v${VERSION}.zip
unzip v${VERSION}.zip
rm v${VERSION}.zip
cp orientation-${VERSION}/data/*quat ./
rm -rf orientation-${VERSION}

# Non-optimal sets
rm c48u9.quat c600vec.quat c48u309.quat c48u527.quat
# Suboptimal sets
rm c48u157.quat c48u519.quat c48u2867.quat c48u4701.quat

python3 quat_to_numpy.py

rm -rf orientation-${VERSION}
rm *.quat
