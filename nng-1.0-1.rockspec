
package = "nnx"
version = "1.0-1"

source = {
   url = "nnx-1.0-1.tgz"
}

description = {
   summary = "A graph extension to Torch7's nn package",
   detailed = [[
         This package provides a graph modeling framework
         for Torch7's nn package.
   ]],
   homepage = "",
   license = "MIT/X11" -- or whatever you like
}

dependencies = {
   "lua >= 5.1",
   "torch"
}

build = {
   type = "cmake",

   variables = {
      CMAKE_INSTALL_PREFIX = "$(PREFIX)"
   }
}
