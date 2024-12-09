# Haskell

$day=$args[0]
$part=$args[1]
$inp=$args[2]

# write-host "Compiling..."
# & 'E:\InstalledPrograms\ghcup\bin\ghc.exe' aoc.hs -fprint-potential-instances

write-host "Running for day $day part $part (on input '$inp')..."
Get-Content .\day${day}_${inp}.txt | cabal run x2024 -- Day${day} Part${part}