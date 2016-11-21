#!/bin/awk -f

$1 % 2 != 0 {
	printf "%d, ", $4;
}

END {
	printf "\n";
}
