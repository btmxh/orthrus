{
  description = "Orthrus dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
          ];

          # PyPI wheels (torch, triton, nvidia-*) expect FHS libs; /run/opengl-driver/lib
          # provides the host driver's libcuda.so.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ] + ":/run/opengl-driver/lib";
          # Triton locates libcuda via `ldconfig -p`, which knows nothing on NixOS.
          TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";

          UV_PYTHON = "${python}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";

          shellHook = ''
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          '';
        };
      }
    );
}
