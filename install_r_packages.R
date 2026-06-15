#!/usr/bin/env Rscript
# install_r_packages.R
# One-time setup: installs required R packages for the Caesar stylometry project.
# Run: Rscript install_r_packages.R

cat("Installing R packages for Caesar Stylometry...\n")

pkgs <- c("stylo", "ggplot2", "reshape2", "cluster", "ape")

for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    cat(sprintf("  %s already installed.\n", pkg))
  }
}

cat("\nAll R packages installed.\n")
cat(sprintf("stylo version: %s\n", as.character(packageVersion("stylo"))))
