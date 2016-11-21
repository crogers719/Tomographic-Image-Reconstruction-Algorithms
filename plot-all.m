#! /usr/bin/octave -q

args = argv();

in = args{1};
out = args{2};

data = load(in);

plot (data(:, 1), data(:, 4));

print(out);
