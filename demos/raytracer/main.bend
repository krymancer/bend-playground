import Base
import ./ray.bend as Ray
import ../shared/grid.bend as G
import ../shared/config.bend as Config

def main() -> IO(Unit):
  do IO<Unit>:
    +width : U32 <- Config.read("RAY_WIDTH", 512)
    +height : U32 <- Config.read("RAY_HEIGHT", 320)
    depth : U32 <- Config.read("TREE_DEPTH", 18)
    bounces : U32 <- Config.read("RAY_BOUNCES", 3)
    +dump : U32 <- Config.read("DUMP", 1)
    start : Nat <- IO.now()
    +pixels : G.Grid = Ray.render!(U32.to_nat(depth), 0, width, height, U32.to_nat(bounces))
    end : Nat <- IO.now()
    IO.print_err("COMPUTE_MS " ++ Nat.show((end - start)))
    IO.print("RAY " ++ U32.show(width) ++ " " ++ U32.show(height) ++ " " ++ U32.show(G.checksum(pixels)))
    IO.print(Bool.pick(String, U32.is_eq(dump, 1), G.text(pixels), ""))
