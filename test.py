class Foo:

    def __len__(self):
        return None


f1, f2, f3 = Foo(), Foo(), Foo()
l = [f1, f2, f3]

print(f1 in l)
