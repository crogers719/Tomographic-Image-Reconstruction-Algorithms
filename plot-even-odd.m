#! /usr/bin/octave -q

args = argv();

in = args{1};
out = args{2};

data = load(in);

plot (data(1:2:end, 1), data(1:2:end, 4), '-xr');
hold on;
plot (data(2:2:end, 1), data(2:2:end, 4),'-xg');
hold off;

print(out);
