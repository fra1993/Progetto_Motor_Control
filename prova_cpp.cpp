#include <iostream>
//#include 

void main() {
	uint16_t x = 553, hi, lo, result;
	printf("%i\n", x);
	hi = (x & 0xff00) >> 8;
	lo = (x & 0x00ff) << 8;
	result = (lo | hi);
	printf("%i\n", result);
}