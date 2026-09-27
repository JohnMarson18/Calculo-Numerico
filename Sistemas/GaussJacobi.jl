using LinearAlgebra

matriz = [
    5.0 -1.0 2.0;
    3.0 8.0 -2.0;
    1.0 1.0 4.0
]
vetor = [12.0, -25.0, 6.0]
chute_inicial = [0.0, 0.0, 0.0]
criterio_erro = 1e-8
limite_iteracoes = 100
exibir_etapas = true

function analisar_linhas(A::AbstractMatrix)
    n = size(A, 1)
    d = abs.(diag(A))
    any(iszero, d) && error("Diagonal principal nao pode conter zeros.")
    alphas = [(sum(abs, @view A[i, :]) - d[i]) / d[i] for i in 1:n]
    return maximum(alphas) < 1.0, maximum(alphas), alphas
end

function resolver_jacobi(A::AbstractMatrix, b::AbstractVector;
    x0=zeros(length(b)), tol=1e-6, max_iter=1000, verbose=true)
    n = length(b)
    size(A) == (n, n) || throw(DimensionMismatch("Matriz e vetor com tamanhos incompativeis."))

    valido, alfa_max, alphas = analisar_linhas(A)
    d = diag(A)

    if verbose
        println("  METODO DE GAUSS-JACOBI")
        println("\n")
        println("  Criterio das Linhas   ", valido ? "Satisfeito" : "Nao satisfeito")
        println("  Fator alpha maximo    $(round(alfa_max, digits=4))")
        println("  Alphas por linha      ", round.(alphas, digits=4))
        println("\n")
        println("Iteracao   Solucao Aproximada x^(k)                  Erro")
        println("\n")
    end

    x = float(copy(x0))
    historico = []
    convergido = false

    for k in 1:max_iter
        x_novo = x + (b - A * x) ./ d
        erro = norm(x_novo - x, Inf) / max(norm(x_novo, Inf), eps())
        push!(historico, (iteracao=k, x=copy(x_novo), erro=erro))
        x = x_novo

        if verbose
            x_str = sprint(show, round.(x, digits=5))
            println(" $(rpad(k, 8))   $(rpad(x_str, 38))   $(round(erro, sigdigits=5))")
        end

        if erro < tol
            convergido = true
            break
        end
    end

    norma_res = norm(b - A * x, Inf)

    if verbose
        println("\n")
        println(" Status final        : ", convergido ? "Convergeu" : "Nao convergiu")
        println(" Total de iteracoes  : $(length(historico))")
        println(" Erro relativo final : $(round(historico[end].erro, sigdigits=5))")
        println(" Norma do residuo    : $(round(norma_res, sigdigits=5))")
        println(" Solucao final x     : ", round.(x, digits=6))
        println("\n")
    end

    return (; x, iteracoes=length(historico), convergido,
        norma_residuo=norma_res, erro_final=historico[end].erro,
        alfa_max, criterio_linhas=valido, alphas, historico)
end

resultado = resolver_jacobi(matriz, vetor; x0=chute_inicial, tol=criterio_erro, max_iter=limite_iteracoes, verbose=exibir_etapas)
@assert resultado.convergido
