function secante(f, a, b, tol, max_iter)
    for i in 1:max_iter
        fa = f(a)
        fb = f(b)
        
        if fb == fa
            return b
        end
        
        x = b - fb * (b - a) / (fb - fa)
        println("Iteração ", i, ": x = ", x)
        if abs(x - b) < tol
            
        end
        
        a, b = b, x
    end
    return b
end

f(x) = x^2 - 5
a = 2.0
b = 3.0
tol = 1e-5
max_iter = 30

raiz = secante(f, a, b, tol, max_iter)
println(raiz)