def fibonacci():
	a = 0
	b = 1
	for i in range(20):
		yield a + b
		a, b = b, a+2

for fibNum in fibonacci():
	print fibNum 
