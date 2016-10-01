phonebook = {
    "John" : 938477566,
    "Jack" : 938377264,
    "Jill" : 947662781
}

for name, number in phonebook.iteritems():
	print "Phone number of %s is %d" % (name, number)

del phonebook["John"]
print "Deleted John"
for name, number in phonebook.iteritems():
        print "Phone number of %s is %d" % (name, number)
