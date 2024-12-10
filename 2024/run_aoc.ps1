# Haskell

$day=$args[0]
$part=$args[1]
$inp=$args[2]

# write-host "Compiling..."
# ghc app/Main.hs -fprint-potential-instances

write-host "Running for day $day part $part (on input '$inp')..."
Get-Content .\day${day}_${inp}.txt | cabal run x2024 -- Day${day} Part${part}
# Get-Content .\day${day}_${inp}.txt |  Day${day} Part${part}