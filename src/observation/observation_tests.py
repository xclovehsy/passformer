from src.observation.autophase import compute_autophase, AUTOPHASE_FEATURE_NAMES
from src.observation.instcount import compute_instcount, INST_COUNT_FEATURE_DIMENSIONALITY
from src.observation.inst2vec import Inst2vecEncoder
import ir2vec
import numpy as np
import torch
import programl as pg
from src.utils.system import read_ir_from_file

def test_autophase(ir_path):
    autophase = compute_autophase(ir_path)
    print(autophase)

def test_instcount(ir_path):
    instcount = compute_instcount(ir_path)
    print(instcount)

def test_inst2vec(ir_path):
    encoder = Inst2vecEncoder()
    text = encoder.preprocess(ir_path)
    encode_text = encoder.encode(text)
    embed_text = encoder.embed(encode_text)
    print(text, encode_text, embed_text)

def test_ir2vec(ir_path):
    initObj = ir2vec.initEmbedding(ir_path, "fa", "p")
    progVector1 = ir2vec.getProgramVector(initObj)
    functionVectorMap1 = ir2vec.getFunctionVectors(initObj)
    instructionVectorsList1 = ir2vec.getInstructionVectors(initObj)
    
    print(progVector1)
    print(np.array(progVector1).shape)
    print([key for key, value in functionVectorMap1.items()])
    print(np.array(instructionVectorsList1).shape)

def test_programl(ir_path):
    ir = read_ir_from_file(ir_path)
    # programl目前只支持llvm版本 3.8.0, 6.0.0, 10.0.0 语法 当前llvm版本14.0.0
    G = pg.from_llvm_ir(ir)
    pg.to_networkx(G)
    pg.save_graphs('file.data', [G])
    

if __name__ == '__main__':
    ir_path = '/home/xucong24/Compiler/tmp/37902.ll'
    # ir_path = '/Users/xucong/Desktop/Compiler/optimized.ll'
    # test_autophase(ir_path)
    # test_instcount(ir_path)
    # test_inst2vec(ir_path)
    test_ir2vec(ir_path)
    # test_programl(ir_path)
    

    
    