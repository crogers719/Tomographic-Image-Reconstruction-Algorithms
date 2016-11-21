#! /usr/bin/octave -q


art_ep = load('art_convergence.txt');
art_t = load('art_time.txt');
sirt_ep = load ('sirt_convergence.txt');
sirt_t = load ('sirt_time.txt');

plot (art_t, art_ep, '-xr')
hold on;
plot (sirt_t, sirt_ep ,'-xg')
hold off;
xlabel ('Time (seconds)');
ylabel ('Epsilon Value');
title('Epsilon vs Time');
legend ('ART' , 'SIRT');

print -dpdf Epsilon-vs-Time.pdf 



