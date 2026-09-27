using LinearAlgebra

matriz = [
    5.0 -1.0 2.0;
    3.0 8.0 -2.0;
    1.0 1.0 4.0
]
vetor = [12.0, -25.0, 6.0]
tolerancia_pivo = 1e-12
exibir_etapas = true

function retro_substituicao(U::AbstractMatrix, c::AbstractVector)
    n = length(c)
    x = zeros(Float64, n)
    for i in n:-1:1
        termo = dot(@view(U[i, i+1:n]), @view(x[i+1:n]))
        x[i] = (c[i] - termo) / U[i, i]
    end
    return x
end

function resolver_gauss(A::AbstractMatrix, b::AbstractVector; tol=1e-12, verbose=true, detalhes=false)
    n = length(b)
    size(A) == (n, n) || throw(DimensionMismatch("A matriz deve ser quadrada e compativel com o vetor."))

    M = float([A b])
    trocas_linhas = 0

    if verbose
        println("  ELIMINACAO GAUSSIANA - PIVOTEAMENTO PARCIAL")
        println("\n")
        println("Matriz aumentada inicial [A | b]:")
        display(round.(M, digits=4))
        println()
    end

    for k in 1:n-1
        p = argmax(abs.(@view M[k:n, k])) + k - 1
        pivo = M[p, k]

        abs(pivo) < tol && error("Pivo nulo na coluna $k.")

        if p != k
            M[[k, p], :] = M[[p, k], :]
            trocas_linhas += 1
            verbose && println("[Passo $k] Troca: L$k <-> L$p | Pivo = $(round(M[k, k], digits=4))")
        else
            verbose && println("[Passo $k] Pivo L$k = $(round(M[k, k], digits=4)) (sem troca)")
        end

        for i in k+1:n
            fator = M[i, k] / M[k, k]
            M[i, k:end] .-= fator .* M[k, k:end]
        end

        if verbose
            println("-> Matriz apos escalonar coluna $k:")
            display(round.(M, digits=4))
            println()
        end
    end

    abs(M[n, n]) < tol && error("Pivo nulo no elemento ($n, $n).")

    x = retro_substituicao(@view(M[:, 1:n]), @view(M[:, n+1]))
    det_A = ((-1)^trocas_linhas) * prod(diag(@view M[:, 1:n]))
    norma_res = norm(b - A * x, Inf)

    if verbose
        println("\n")
        println("  RESULTADOS FINAIS")
        println("\n")
        println("  • Trocas de linha  : $trocas_linhas")
        println("  • det(A) calculado : $(round(det_A, digits=6))")
        println("  • Norma residuo    : $(round(norma_res, sigdigits=4))")
        println("  • Solucao x        : ", round.(x, digits=6))
        println("\n")
    end

    return detalhes ? (; x, determinante=det_A, trocas_linhas, norma_residuo=norma_res) : x
end

solucao = resolver_gauss(matriz, vetor; tol=tolerancia_pivo, verbose=exibir_etapas)
@assert norm(matriz * solucao - vetor) < 1e-10
