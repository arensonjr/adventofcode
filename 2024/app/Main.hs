{-# LANGUAGE ScopedTypeVariables, TypeSynonymInstances, FlexibleInstances, FlexibleContexts #-}

module Main where

import Control.Exception (IOException, catch)
import Data.List (sort, findIndices, elemIndices, elemIndex, sortBy, isInfixOf, transpose, isPrefixOf, isSuffixOf)
import Data.List.Split (splitOn)
import Data.Char (digitToInt)
import Data.Ord (Down(Down), comparing, Ordering(LT, EQ, GT))
import Data.Maybe (catMaybes, isJust, isNothing, listToMaybe, fromMaybe)
import Debug.Trace (trace)
import System.Environment (getArgs)
import System.IO (isEOF)
import Text.Read (readMaybe)

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

step :: (Show a) => String -> a -> a
step tag val = if debug then trace (tag ++ ": {{ " ++ show val ++ " }}") val else val

findAll :: Regex -> String -> [[String]]
findAll pattern input =
    case matchRegexAll pattern input of
        Nothing -> []
        Just (_, _, after, matches) -> matches : findAll pattern after

-- "<>" is "`mappend`", the monoid "add":
-- rotate n xs = drop n xs ++ take n xs
-- rotate n = drop n <> take n
rotate :: Int -> [a] -> [a]
rotate =  drop <> take

getLines :: IO [[String]]
getLines = do
    done <- isEOF
    if done then return [] else do
        line <- getLine `catch` \(e :: IOException) -> return []
        (:) (parseLine line) <$> getLines

parseLine ls = filter (not . null) . words $ ls

main :: IO ()
main = getArgs >>= runMain >>= print
runMain [day, part] = runDay (read day) (read part) <$> getLines

--------------------------------------------------------------

day1 :: Part -> [[String]] -> Int
parseDay1 idx = sort . map (idx . map read)
day1 Part1 nums = sum $ zipWith (\x y -> abs (x - y)) (parseDay1 head nums) (parseDay1 last nums)
day1 Part2 nums =
    let right = parseDay1 last nums
    in sum [x * (length . elemIndices x) right | x <- parseDay1 head nums]

day2 :: Part -> [[String]] -> Int
incOrDec xs = (xs == sort xs) || (xs == sortBy (comparing Data.Ord.Down) xs)
smallDelta (x:y:rest) = (x /= y) && (abs (x-y) <= 3) && (smallDelta (y:rest))
smallDelta _ = True

day2 Part1 = length . filter incOrDec . filter smallDelta . map (map read)
day2 Part2 = length . filter (not . null). map (filter incOrDec . filter smallDelta . day2options . map read)

day2options xs = [take n xs ++ drop (n+1) xs | n <- [0..(length xs - 1)]]

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

between :: (Num a, Ord a) => a -> a -> a -> Bool
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
    let candidates = extend pos vector (fromInteger (-1)) size
    let nextFound = listToMaybe $ sortBy (comparing (taxiDistance pos)) (L.intersect candidates boxes)
    case nextFound of
       Nothing -> Nothing
       -- Conditional necessary because for part 2, standing in place counts as having looped (which is not what I intended)
       Just next -> if next == pos + vector then Nothing else
                    let newBoxes = extend pos vector (pos + vector) (next - vector)
                    in Just (next - vector, S.fromList newBoxes, rightTurn vector)

walk visited boxes pos size vector = do
    case nextBox size boxes pos vector of
        Just (nextPos, newVisited, newVector) -> walk (S.union visited newVisited) boxes nextPos size newVector
        Nothing -> S.union visited . S.fromList $ extend pos vector (fromInteger (-1)) size

day6 Part1 input = do
    let (startPos, boxes, size) = parseDay6 input
    let vector = (0, -1) -- up
    let visited = walk (S.fromList [startPos]) boxes startPos size vector
    length visited

day6 Part2 input = do
    let (startPos, boxes, size@(sizeX, sizeY)) = parseDay6 input
    let vector = (0, -1) -- up
    let candidateBoxes = [(x, y) | x <- [0..sizeX-1], y <- [0..sizeY-1],
                                  not ((x, y) `elem` boxes)]
    let results = map (\b -> (b, tryWalk [] ((step "trying box" b):boxes) startPos size vector)) $ candidateBoxes
    (length . S.fromList . step "final obstacles") [pos | (pos, Just True) <- results]

tryWalk visited boxes pos size vector = do
    (nextPos, _, newVector) <- nextBox size boxes pos vector
    if nextPos `elem` visited
        then return True
        else tryWalk (nextPos:visited) boxes nextPos size newVector

day7 :: Part -> [[String]] -> Int
parseDay7Line :: [String] -> (Int, [Int])
parseDay7Line line = step "parsedLine" $ (((head & init & read) line), (map read . tail) line)

computableAM target [x] = step ("is " ++ show target ++ " => " ++ show [x] ++ " equal?") $
    target == x
computableAM target (x:rest) = step ("is " ++ show target ++ " => " ++ show (x:rest) ++ " computable?") $
    computableAM (target - x) rest
    || (remainder == 0 && computableAM quotient rest)
    where (quotient, remainder) = (target `divMod` x)

day7 Part1 = map parseDay7Line & filter computable & map fst & sum
    where computable (target, nums) = computableAM target (reverse nums)

day7 Part2 = map parseDay7Line & filter computable & map fst & sum
    where computable (target, nums) = computableAMC target (reverse nums)

computableAMC target [x] = step ("is " ++ show target ++ " => " ++ show [x] ++ " equal?") $
    target == x
computableAMC target (x:rest) = step ("is " ++ show target ++ " => " ++ show (x:rest) ++ " computable?") $
    computableAMC (target - x) rest
    || (show x `isSuffixOf` show target) && computableAMC truncated rest
    || (remainder == 0 && computableAMC quotient rest)
    where (quotient, remainder) = (target `divMod` x)
          truncated = (show & take (length (show target) - length (show x)) & step "trying to read" & readMaybe & fromMaybe 0) target

day8 :: Part -> [[String]] -> Int
parseDay8 :: [[String]] -> (MM.MultiMap Char Coord, Coord)
parseDay8 lines = do
    let grid = map head lines
    let antennae = MM.fromList [(val, (x, y)) | (y, row) <- zip [0..] grid, (x, val) <- zip [0..] row, val /= '.']
    (antennae, (length (head grid), length grid))

findInterference :: [Coord] -> [Coord]
findInterference coords = step "interfering" $ L.concat [interfere c1 c2 | c1 <- coords, c2 <- coords, c1 /= c2]
    where interfere one two = let delta = (two - one) in [one - delta, two + delta]

validLocation :: Coord -> Coord -> Bool
validLocation (sizeX, sizeY) (x, y) = x >= 0 && y >= 0 && x < sizeX && y < sizeY

day8 Part1 input = do
    let (antennae, size) = parseDay8 input
    let locs = (MM.elems & map findInterference & L.concat & filter (validLocation (step "size" size)) & L.nub) antennae
    length (step "locs" locs)

day8 Part2 input = do
    let (antennae, size) = parseDay8 input
    let locs = (MM.elems & map (findInterferencePt2 size) & L.concat & L.nub) antennae
    length (step "locs" locs)

findInterferencePt2 :: Coord -> [Coord] -> [Coord]
findInterferencePt2 size coords = step "interfering" $ L.concat [interferePt2 size c1 c2 | c1 <- coords, c2 <- coords, c1 /= c2]
interferePt2 :: Coord -> Coord -> Coord -> [Coord]
interferePt2 size one two = do
    let delta = antennaStep one two
    let prune = takeWhile (validLocation size)
    let go x dx = if validLocation size x then (x:go (x+dx) dx) else []
    go one delta ++ go one (-delta)
antennaStep :: Coord -> Coord -> Coord
antennaStep (x1, y1) (x2, y2) = do
    let dx = x2 - x1
    let dy = y2 - y1
    let denom = gcd dx dy
    (dx `div` denom, dy `div` denom)

day9 :: Part -> [[String]] -> Int
data Block = File Int Int Int | Free Int Int deriving (Show, Read, Eq)
parseDay9 :: [[String]] -> ([Block], [Block])
-- parseDay9 input = ([File 1 0 5, File 2 10 3], [Free 5 6])
parseDay9 input = parseDay9_ 0 0 (input !! 0 !! 0)
parseDay9_ nextFileId pos [lastFile] = ([File nextFileId pos (digitToInt lastFile)], [])
parseDay9_ nextFileId pos (fileStr:freeStr:rest) =
    let file = digitToInt fileStr
        free = digitToInt freeStr
        (files, frees) = parseDay9_ (nextFileId + 1) (pos + file + free) rest
    in (File nextFileId pos file : files, (Free (pos+file) free:frees))

fragment files [] = files
fragment allFiles@(File fileId fileStart fileSize:files) allFrees@(Free freeStart freeSize:frees) = do
    if fileStart < freeStart
        then allFiles
    else if fileSize < freeSize
        then fragment (files ++ [File fileId freeStart fileSize]) (Free (freeStart+fileSize) (freeSize-fileSize):frees)
    else if fileSize == freeSize
        then fragment (files ++ [File fileId freeStart fileSize]) frees
    else -- if fileSize > freeSize
        fragment (File fileId fileStart (fileSize-freeSize):files ++ [File fileId freeStart freeSize]) frees

checksum total [] = total
checksum soFar (File fileId start size:files) = checksum (soFar + fileChecksum) files
    where fileChecksum = sum (zipWith (*) [fileId, fileId..] [start..start+size-1])

day9 Part1 input = do
    let (files, free) = parseDay9 input
    checksum 0 $ fragment (reverse files) free

day9 Part2 input = do
    let (files, free) = step "parsed" $ parseDay9 input
    checksum 0 $ defrag [] (reverse files) free

defrag outFiles [] _ = outFiles
defrag outFiles (file:files) frees =
    let (newFile, newFrees) = fileFit file frees
    in defrag (newFile:outFiles) files newFrees

-- Invariant: Frees stay sorted in ascending order so once we exceed fileStart<freeStart we're done looking
fileFit file [] = (file, [])
fileFit file@(File fileId fileStart fileSize) allFrees@(free@(Free freeStart freeSize):frees) =
    if fileStart < freeStart
        then (file, allFrees)
    else if fileSize < freeSize
        then (File fileId freeStart fileSize, (Free (freeStart+fileSize) (freeSize-fileSize)):frees)
    else if fileSize == freeSize
        then (File fileId freeStart fileSize, frees)
    else -- if fileSize > freeSize
        let (newFile, newFrees) = fileFit file frees in (newFile, free:newFrees)
    
day10 :: Part -> [[String]] -> Int
parseDay10 = step "parsed input" . M.map digitToInt . indexify . map head
neighbors (x, y) = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

dfs10 :: M.Map Coord Int -> Int -> Coord -> [Coord]
dfs10 grid 9 coord = [coord]
dfs10 grid val coord = do
    let nbrs = step ("found neighbors of " ++ show val ++ " at " ++ show coord) $ filter (flip M.lookup grid & (Just (val+1)==)) (neighbors coord)
    L.concat $ map (dfs10 grid (val+1)) nbrs

day10 Part1 input = do
    let grid = parseDay10 input
    let zeros = step "found zeros" [coord | coord <- M.keys grid, M.lookup coord grid == Just 0]
    let nines = map (dfs10 grid 0 & step "found nines" & L.nub & length) zeros
    sum nines

day10 Part2 input = do
    let grid = parseDay10 input
    let zeros = step "found zeros" [coord | coord <- M.keys grid, M.lookup coord grid == Just 0]
    let nines = map (dfs10 grid 0 & step "found nines" & length) zeros
    sum nines

day11 Part1 input = 0
day12 Part1 input = 0
day13 Part1 input = 0
day14 Part1 input = 0
day15 Part1 input = 0
day16 Part1 input = 0
day17 Part1 input = 0
day18 Part1 input = 0
day19 Part1 input = 0
day20 Part1 input = 0
day21 Part1 input = 0
day22 Part1 input = 0
day23 Part1 input = 0
day24 Part1 input = 0
day25 Part1 input = 0

debug :: Bool
debug = False