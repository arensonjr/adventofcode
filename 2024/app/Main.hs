{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Exception (IOException, catch)
import Data.List (sort, findIndices, elemIndices, sortBy, isInfixOf, transpose)
import Data.List.Split (splitOn)
import Data.Ord (Down(Down), comparing)
import Data.Maybe (catMaybes)
import Debug.Trace (trace)
import System.Environment (getArgs)

import Text.Regex (Regex, mkRegex, matchRegex, matchRegexAll)

import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.DiGraph as G

(%) :: (a ->b) -> (b -> c) -> a -> c
(%) = flip (.)
infixr 9 %

data Day = Day1 | Day2 | Day3 | Day4 | Day5
         | Day6 | Day7 | Day8 | Day9 | Day10
         | Day11 | Day12 | Day13 | Day14 | Day15
         | Day16 | Day17 | Day18 | Day19 | Day20
         | Day21 | Day22 | Day23 | Day24 | Day25
         deriving (Read, Show)

data Part = Part1 | Part2 deriving (Read, Show)

readDay :: String -> Day
readDay = read

readPart :: String -> Part
readPart = read

runDay :: Day -> Part -> [[String]] -> Int
runDay Day1  = day1 
runDay Day2  = day2 
runDay Day3  = day3 
runDay Day4  = day4 
runDay Day5  = day5 
runDay Day6  = day6 
runDay Day7  = day7 
runDay Day8  = day8 
runDay Day9  = day9 
runDay Day10 = day10
runDay Day11 = day11
runDay Day12 = day12
runDay Day13 = day13
runDay Day14 = day14
runDay Day15 = day15
runDay Day16 = day16
runDay Day17 = day17
runDay Day18 = day18
runDay Day19 = day19
runDay Day20 = day20
runDay Day21 = day21
runDay Day22 = day22
runDay Day23 = day23
runDay Day24 = day24
runDay Day25 = day25

step :: (Show a) => String -> a -> a
-- DEBUG:
step tag val = trace (tag ++ ": {{ " ++ show val ++ " }}") val 
-- NO DEBUG:
-- step tag val = val

getLines :: IO [[String]]
getLines = do
    line <- getLine `catch` \(e :: IOException) -> return []
--    case line of
--        [] -> return []
--        ls -> fmap ((:) . parseLine $ line) getLines
    (:) (parseLine line) <$> getLines

parseLine :: String -> [String]
parseLine [] = []
parseLine ls = filter (not . null) . words $ ls

main :: IO ()
main = do
    args <- getArgs
    let day = (readDay . head) args
    let part = (readPart . head . drop 1) args
    lines <- getLines
    putStrLn . show . runDay day part $ lines

--------------------------------------------------------------

day1 :: Part -> [[String]] -> Int
-- Part 1
day1 Part1 nums = do
    let left = step "left" (sort . map (head . map read) $ nums)
    let right = step "right" (sort . map (last . map read) $ nums)
    sum $ zipWith (\l r -> abs (l - r)) left right
-- Part 2
day1 Part2 nums = do
    let left = step "left" (sort . map (head . map read) $ nums)
    let right = step "right" (sort . map (last . map read) $ nums)
    let counts = map (\x -> length . elemIndices x $ right) left 
    sum [x * y | (x, y) <- zip left counts]

day2 :: Part -> [[String]] -> Int
-- Part 1
incOrDec xs = (xs == sort xs) || (xs == sortBy (comparing Data.Ord.Down) xs)
smallDelta [] = True
smallDelta [_] = True
smallDelta (x:y:rest) = x /= y && abs (x-y) <= 3 && smallDelta (y:rest)
day2 Part1 nums = length . filter incOrDec . filter smallDelta . map (map read) $ nums
-- Part 2
day2 Part2 nums = length . filter (not . null). map (filter incOrDec . filter smallDelta . day2options . map read) $ nums
day2options xs = [take n xs ++ drop (n+1) xs | n <- [0..(length xs - 1)]]


findAll :: Regex -> String -> [[String]]
findAll pattern input =
    case matchRegexAll pattern input of
        Nothing -> []
        Just (_, _, after, matches) -> matches : findAll pattern after

day3 :: Part -> [[String]] -> Int
mulRegex = mkRegex "mul\\(([0-9]+),([0-9]+)\\)"
mulLine line = sum [read x * read y | [x, y] <- findAll mulRegex line]
day3 Part1 gobbledigook = sum . step "line sums" . map (mulLine . concat) . step "input" $ gobbledigook

day3 Part2 gobbledigook = maybeMul True . concat . concat $ gobbledigook
maybeMulRegex = mkRegex "(do\\(\\))|(don't\\(\\))|(mul\\(([0-9]+),([0-9]+)\\))"
maybeMul additionEnabled line =
    case matchRegexAll maybeMulRegex line of
        Nothing -> 0
        Just (_, _, after, ["", "don't()", "", "", ""]) -> maybeMul False after
        Just (_, _, after, ["do()", "", "", "", ""]) -> maybeMul True after
        Just (_, _, after, ["", "", mul, x, y]) -> (if additionEnabled then read x * read y else 0) + maybeMul additionEnabled after
        Just (_, _, after, matches) -> trace ("unrecognized case: " ++ show matches) 0

-- "<>" is "`mappend`", the monoid "add":
-- rotate n xs = drop n xs ++ take n xs
-- rotate n = drop n <> take n
rotate =  drop <> take

day4 :: Part -> [[String]] -> Int
findXmas = length . findAll (mkRegex "(XMAS)") 
diagonalize :: [[a]] -> [[a]]
diagonalize = (transpose . diagonalizeTop_) <> (drop 1 . transpose . diagonalizeBot_)
diagonalizeTop_ [] = []
diagonalizeTop_ grid = head grid : (diagonalizeTop_ . map (drop 1)) (tail grid)
diagonalizeBot_ = reverse . diagonalizeTop_ . reverse
day4 Part1 wordsearch = do
    let ltor = step "ltor" $ map head wordsearch
    let rtol = step "rtol" $ map reverse ltor
    let utod = step "utod" $ transpose ltor
    let dtou = step "dtou" $ map reverse utod
    let diagdr = step "diagdr" $ diagonalize ltor
    let diagur = step "diagur" $ map reverse diagdr
    let diagdl = step "diagdl" $ diagonalize rtol
    let diagul = step "diagul" $ map reverse diagdl
    (sum . map findXmas) (concat [ltor, rtol, utod, dtou, diagdr, diagdl, diagur, diagul])

day4 Part2 wordsearch = do
    let grid = indexify (map head wordsearch)
    let middleAs = [(x, y) | ((x, y), val) <- M.toList grid,
                             val == 'A',
                             x > 0, x < length grid - 1,
                             y > 0, y < length grid - 1]
    let found = map (isXMas grid) middleAs
    length $ filter id found
indexify grid = M.fromList [((x, y), val) | (y, vals) <- zip [0..] grid, (x, val) <- zip [0..] vals]
surrounding grid (x, y) = do
    let aroundIndices = [(x-1, y-1), (x+1, y-1), (x+1, y+1), (x-1, y+1)]
    catMaybes [M.lookup coords grid | coords <- aroundIndices]
-- Sort/equal is not the right way to do this. Hmm
--isXMas grid = ("MMSS" ==) . sort . surrounding grid
isXMas grid coord = do
    let surround = surrounding grid coord
    any (=="MMSS") $ map (flip rotate surround) [0..3]

day5 :: Part -> [[String]] -> Int
parseDay5 :: [[String]] -> (G.DiGraph Int, [[Int]])
parseDay5 input = do
    let inp = step "inp" $ map head (step "input" input)
    let edges = step "edges unparsed" $ takeWhile isEdge inp
    let lists = step "lists unparsed" $ drop (length edges + 1) (step "inp" inp)
    (G.fromEdges (map parseDay5Edge edges), map parseDay5List lists)

isEdge = elem '|'
--parseDay5Edge line = let edge = head line in ((read . take 2) edge, (read . drop 3) edge)
parseDay5Edge = (\[x, y] -> (x, y)) . map read . splitOn "|"
parseDay5List = map read . splitOn "," . step "lists"

day5 Part1 nums = do
    let (graph, ls) = step "parsed" $ parseDay5 nums
    length ls
day5 Part2 nums = 0


day6  part nums = 0
day7  part nums = 0
day8  part nums = 0
day9  part nums = 0
day10 part nums = 0
day11 part nums = 0
day12 part nums = 0
day13 part nums = 0
day14 part nums = 0
day15 part nums = 0
day16 part nums = 0
day17 part nums = 0
day18 part nums = 0
day19 part nums = 0
day20 part nums = 0
day21 part nums = 0
day22 part nums = 0
day23 part nums = 0
day24 part nums = 0
day25 part nums = 0