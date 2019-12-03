#!/bin/sh

GRMFILE=$1
NAME=$2

grmtest $GRMFILE.grm $NAME; far2fst $GRMFILE.far; fstinfo $NAME.fst; ./fst2pdf $NAME.fst
