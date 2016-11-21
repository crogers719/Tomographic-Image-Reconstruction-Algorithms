#!/bin/bash

kernel=$1
logfile=$1.log
program="sirt-prof-$1.exe"

rm -f $logfile
for x in 192 96 64 32
do
	for y in {1..6} $(seq 8 2 32)
	do
		for z in {1..6} $(seq 8 2 32)
		do
			if [ $(($x * $y * $z)) -gt 1024 ]
			then
				break
			else
				echo $x $y $z
				echo $x $y $z >> $logfile
				nvprof ./$program fo-1.txt 10 26 0 1 0 0 0 0 $x $y $z 2>&1 | awk -v k="$kernel\\\(" '$0 ~ k{ print $2, $4, $5, $6; }' >> $logfile
			fi
		done
	done
done
