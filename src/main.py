from .array import NumpyArray

if __name__ == "__main__":
    a = NumpyArray([1, 2, 3])
    b = NumpyArray([10, 20, 30])

    print(a + b)          
    print(a + 100)        
    print(100 + a)        
    print(b - a)         
    print(100 - a)        