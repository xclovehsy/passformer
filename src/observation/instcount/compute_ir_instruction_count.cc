#include <iostream>

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

#include "InstCount.h"

using namespace instcount;

int main(int argc, char** argv) {

  if (argc != 2) {
        std::cerr <<  "Usage: compute_autophase <bitcode-path>";
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
  // CHECK(module) << "Failed to parse: " << argv[1] << ": " << error.getMessage().str();

  // std::cout << module->getInstructionCount() << std::endl;

  InstCountFeatureVector features = InstCount::getFeatureVector(*module);
  for (size_t i = 0; i < features.size(); ++i) {
    if (i) {
      std::cout << " ";
    }
    std::cout << features[i];
  }
  std::cout << std::endl;
  

  return 0;
}
