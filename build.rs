fn main() {
    // Register the has_cuda cfg to avoid unexpected_cfgs warnings
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");
    
    // assume CUDA is available on linux
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-cfg=has_cuda");
        println!("cargo:warning=Linux detected - enabling CUDA features");
    } else {
        println!("cargo:warning=Not on Linux - CUDA features will be disabled");
    }
    
    // Fix GLib/GObject library paths on macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-search=/opt/homebrew/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/homebrew/lib");
    }
}