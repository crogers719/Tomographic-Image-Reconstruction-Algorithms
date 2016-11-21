#! /usr/bin/octave -q

args = argv();
fname = args{1};
f = load(fname);

figure(1);
imagesc(f);

[_, name, _] = fileparts(fname);

print(name, '-dpng')
