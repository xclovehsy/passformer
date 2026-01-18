#include <iostream>

#include "InstCount.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

int main(int argc, char** argv) {
  if (argc != 2) {
        std::cerr << "Usage: compute_autophase <bitcode-path>";
        return 1;
    }

  auto buf = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (!buf) {
      std::cerr << "Error reading LLVM IR file\n";
      return 1;
  }

  llvm::SMDiagnostic error;
  llvm::LLVMContext ctx;

  auto module = llvm::parseIRFile(argv[1], error, ctx);
  
  // Print feature vector to stdout.
  const auto features = autophase::InstCount::getFeatureVector(*module);
  for (size_t i = 0; i < features.size(); ++i) {
    if (i) {
      std::cout << " ";
    }
    std::cout << features[i];
  }
  std::cout << std::endl;

  return 0;
}
