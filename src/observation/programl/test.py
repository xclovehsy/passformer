import programl as pg


with open('/root/Compiler/optimized.ll', 'r') as f:
    llvm_ir = f.read()

print(llvm_ir)

G = pg.from_llvm_ir(llvm_ir)
pg.to_networkx(G)
pg.save_graphs('./file.data', [G])