{-# LANGUAGE ScopedTypeVariables, TypeSynonymInstances, FlexibleInstances #-}

module Main where

import Control.Exception (IOException, catch)
import Data.List (sort, findIndices, elemIndices, elemIndex, sortBy, isInfixOf, transpose)
import Data.List.Split (splitOn)
import Data.Ord (Down(Down), comparing, Ordering(LT, EQ, GT))
import Data.Maybe (catMaybes, isJust, isNothing, listToMaybe)
import Debug.Trace (trace)
import System.Environment (getArgs)
import System.IO (isEOF)

import Text.Regex (Regex, mkRegex, matchRegex, matchRegexAll)

import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.MultiMap as MM
import qualified Data.Set as S
import qualified Data.HashSet as HS
import qualified Data.DiGraph as G

-- I wonder if I'd use this...
(&) :: (a ->b) -> (b -> c) -> a -> c
(&) = flip (.)
infixl 9 &

data Day = Day1 | Day2 | Day3 | Day4 | Day5
         | Day6 | Day7 | Day8 | Day9 | Day10
         | Day11 | Day12 | Day13 | Day14 | Day15
         | Day16 | Day17 | Day18 | Day19 | Day20
         | Day21 | Day22 | Day23 | Day24 | Day25
         deriving (Read, Show)

data Part = Part1 | Part2 deriving (Read, Show)

type Coord = (Int, Int)
instance Num Coord where
    (x, y) + (a, b) = (x+a, y+b)
    (x, y) * (a, b) = (x*a, y*b)
    negate (x, y) = (-x, -y)
    fromInteger x = (fromInteger x, fromInteger x)
    signum (x, y) = (signum x, signum y)
    abs (x, y) = (abs x, abs y)
taxiDistance :: Coord -> Coord -> Int
taxiDistance src dest = dx + dy
    where (dx, dy) = abs (dest - src)



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
-- step tag val = trace (tag ++ ": {{ " ++ show val ++ " }}") val 
-- NO DEBUG:
step tag val = val

getLines :: IO [[String]]
getLines = do
    done <- isEOF
    if done then return [] else do
        line <- getLine `catch` \(e :: IOException) -> return []
        (:) (parseLine line) <$> getLines

parseLine ls = filter (not . null) . words $ ls

main :: IO ()
main = do
    args <- getArgs
    let day = readDay (head args)
    let part = readPart (args !! 1)
    lines <- getLines
    print . runDay day part $ lines

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
isXMas grid coord = do
    let surround = surrounding grid coord
    elem "MMSS" . map (`rotate` surround) $ [0..3]

day5 :: Part -> [[String]] -> Int
parseDay5 input = do
    let [edges, lists] = step "split lits" [map head ls | ls <- splitOn [[]] input]
    (MM.fromList (map parseDay5Edge edges), map parseDay5List lists)
parseDay5Edge = (\[x, y] -> (x, y)) . map read . splitOn "|"
parseDay5List = step "lists" & splitOn "," & map read
checkDay5List graph ls = (not . or) [elemIndex x ls > elemIndex y ls | (x, y) <- MM.toList graph, x `elem` ls, y `elem` ls]
graphOrdering graph x y
  | y `elem` MM.lookup x graph = LT
  | x `elem` MM.lookup y graph = GT
  | otherwise = EQ
mid ls = ls !! (length ls `div` 2)

day5 Part1 nums = do
    let (graph, lists) = parseDay5 nums
    let withLens = zip lists (map length lists)
    let mids = step "mids" $ [ls !! (len `div` 2) | (ls, len) <- withLens]
    let checked = step "checked" $ map (checkDay5List graph) lists
    sum [mid | (mid, check) <- zip mids checked, check]

day5 Part2 nums = do
    let (graph, lists) = parseDay5 nums
    let checked = step "checked" $ map (checkDay5List graph) lists
    let nonsorted = [list | (list, check) <- zip lists checked, not check]
    let sorted = map (sortBy (graphOrdering graph)) nonsorted
    let mids = step "mids" $ [ls !! (length ls `div` 2) | ls <- sorted]
    sum mids



day6 :: Part -> [[String]] -> Int

rightTurn :: (Int, Int) -> (Int, Int)
rightTurn (0, y) = (negate y, 0)
rightTurn (x, 0) = (0, x)

between :: Int -> Int -> Int -> Bool
between bound1 mid bound2
    | bound1 < mid = mid < bound2
    | bound1 > mid = mid > bound2
    | otherwise = False

extendOne :: Int -> Int -> Int -> Int -> [Int]
extendOne x 0 _ _ = [x]
extendOne x dx bound1 bound2 = [x, x+dx .. boundX-dx]
    where boundX = if dx < 0 then min bound1 bound2 else max bound1 bound2
extend :: Coord -> Coord -> Coord -> Coord -> [Coord]
extend (x, y) (vecX, vecY) (startX, startY) (endX, endY) = [
    (x', y') | x' <- extendOne x vecX startX endX,
               y' <- extendOne y vecY startY endY]

parseDay6 :: [[String]] -> (Coord, [Coord], Coord)
parseDay6 lines = do
    let grid = map head lines
    let size = (length (head grid), length grid)
    let withCoords = [(x, y, val) | (y, row) <- zip [0..] grid, (x, val) <- zip [0..] row]
    let startPos = head [(x, y) | (x, y, val) <- withCoords, val == '^']
    let boxes = [(x, y) | (x, y, val) <- withCoords, val == '#']
    (startPos, boxes, size)

nextBox size boxes pos vector = do
    let candidates = step "candidates" $ extend pos vector (fromInteger (-1)) size
    let found = step "found" $ L.intersect candidates boxes
    let nextFound = step "nextFound" $ listToMaybe $ sortBy (comparing (taxiDistance pos)) found
    case nextFound of
       Nothing -> step "next found nothing" Nothing
       Just next -> do
            let newBoxes = step "newBoxes" $ extend pos vector (pos + vector) (next - vector)
            let newVector = step "newVector" $ rightTurn vector
            Just (next - vector, S.fromList newBoxes, newVector)

walk visited boxes pos size vector = do
    case nextBox size boxes pos vector of
        Just (nextPos, newVisited, newVector) -> walk (S.union visited newVisited) boxes nextPos size newVector
        Nothing -> S.union visited . S.fromList $ extend pos vector (fromInteger (-1)) size

day6 Part1 input = do
    let (startPos, boxes, size) = step "parsed" $ parseDay6 input
    let vector = (0, -1) -- up
    let visited = step "final visited" $ walk (S.fromList [startPos]) boxes startPos size vector
    length visited

day6 Part2 input = do
    let (startPos, boxes, size) = step "parsed" $ parseDay6 input
    let vector = (0, -1) -- up
    let corners = step "final corners" $ walkCorners [] boxes startPos size vector
    let obstacles = step "obstacles" $ setObstacles boxes (startPos:reverse corners)
    (length . S.fromList) obstacles

nextBoxCorners size boxes pos vector = do
    let candidates = step "candidates" $ extend pos vector (fromInteger (-1)) size
    let found = step "found" $ L.intersect candidates boxes
    let nextFound = step "nextFound" $ listToMaybe $ sortBy (comparing (taxiDistance pos)) found
    case nextFound of
       Nothing -> step "next found nothing" Nothing
       Just next -> do
            let newVector = step "newVector" $ rightTurn vector
            Just (next - vector, newVector)
walkCorners corners boxes pos size vector = do
    case nextBoxCorners size boxes pos vector of
        Just (nextPos, newVector) -> walkCorners (nextPos:corners) boxes nextPos size newVector
        Nothing -> corners

-- TODO: This doesn't work and not sure why. Probably time to just go brute-force and try each new box location to see if I
-- re-cross the same corner before exiting the grid.
setObstacles :: [Coord] -> [Coord] -> [Coord]
setObstacles boxes (x:y:z:rest) = do
    -- TODO: Technically, I should account for the obstacles being 1 step farther
    -- than where the guard stops, but I don't think it'll matter...
    -- TODO: Need to handle overshoots: the actual obstacle was sooner than the theoretical obstacle would have to be
    -- TODO: Probably need to handle when the theoretical obstacle would have blocked a traversal earlier in the path
    let obstacle = step "newObstacle" $ findObstacle (step "x" x) (step "y" y) (step "z" z)
    let onTheWay = step "onTheWay" $ extend z (signum (obstacle - z)) z obstacle
    let remainder = step "remainder" $ setObstacles boxes (y:z:rest)
    if null (step "intersected" $ L.intersect onTheWay boxes)
        then obstacle : remainder
        else remainder
setObstacles _ tooFewCorners = []

findObstacle (x1, y1) (x2, y2) (x3, y3)
    | x1 == x2 = (x3, y1)
    | y1 == y2 = (x1, y3)
    | otherwise = error "WAT"

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