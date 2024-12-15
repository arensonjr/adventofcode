$day=$args[0]
$part=$args[1]
$inp=$args[2]

if ([string]::IsNullOrWhiteSpace($args[3])) {
    $debug="False"
} else {
    $debug="True"
}

########## Haskell

# (not necessary with `cabal run`)
# write-host "Compiling..."
# ghc app/Main.hs -fprint-potential-instances

# write-host "Running for day $day part $part (on input '$inp')..."
# Get-Content .\day${day}_${inp}.txt | cabal run x2024 -- Day${day} Part${part}

########## Python
# uv run -m cProfile aoc.py "day${day}" "part${part}" "day${day}_${inp}.txt" $debug
uv run aoc.py "day${day}" "part${part}" "day${day}_${inp}.txt" $debug

########## Kotlin
# write-host "Compiling..."
# kotlinc aoc.kt
# write-host "Running..."
# kotlin AocKt day${day} part${part} "day${day}_${inp}.txt" $debug

########## Rust
# if ([string]::IsNullOrWhiteSpace($args[3])) {
#     cargo run -- "day${day}" "part${part}" "day${day}_${inp}.txt" $debug
# } else {
#     cargo run --release -- "day${day}" "part${part}" "day${day}_${inp}.txt" $debug
# }
