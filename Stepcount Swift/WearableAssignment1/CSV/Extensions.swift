//
//  Extensions.swift
//  WearableAssignment1
//
//  Created by Benedikt Langer on 11.12.21.
//

import Accelerate
import UIKit

extension ViewController {
    static func solveLinearSystem(a: inout [Double],
                                  a_rowCount: Int, a_columnCount: Int,
                                  b: inout [Double],
                                  b_count: Int) throws {
        
        var info = Int32(0)
        
        // 1: Specify transpose.
        var trans = Int8("T".utf8.first!)
        
        // 2: Define constants.
        var m = __CLPK_integer(a_rowCount)
        var n = __CLPK_integer(a_columnCount)
        var lda = __CLPK_integer(a_rowCount)
        var nrhs = __CLPK_integer(1) // assumes `b` is a column matrix
        var ldb = __CLPK_integer(b_count)
        
        // 3: Workspace query.
        var workDimension = Double(0)
        var minusOne = Int32(-1)
        
        dgels_(&trans, &m, &n,
               &nrhs,
               &a, &lda,
               &b, &ldb,
               &workDimension, &minusOne,
               &info)
        
        if info != 0 {
            throw LAPACKError.internalError
        }
        
        // 4: Create workspace.
        var lwork = Int32(workDimension)
        var workspace = [Double](repeating: 0,
                                 count: Int(workDimension))
        
        // 5: Solve linear system.
        dgels_(&trans, &m, &n,
               &nrhs,
               &a, &lda,
               &b, &ldb,
               &workspace, &lwork,
               &info)
        
        if info < 0 {
            throw LAPACKError.parameterHasIllegalValue(parameterIndex: abs(Int(info)))
        } else if info > 0 {
            throw LAPACKError.diagonalElementOfTriangularFactorIsZero(index: Int(info))
        }
    }
}

public enum LAPACKError: Swift.Error {
    case internalError
    case parameterHasIllegalValue(parameterIndex: Int)
    case diagonalElementOfTriangularFactorIsZero(index: Int)
}
