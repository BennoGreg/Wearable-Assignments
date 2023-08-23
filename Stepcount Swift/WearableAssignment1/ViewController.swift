//
//  ViewController.swift
//  WearableAssignment1
//
//  Created by Benedikt Langer on 02.12.21.
//

import UIKit
import SensorKit
import Accelerate
import CoreMotion
import simd

class ViewController: UIViewController {
    
    let motion = CMMotionManager()
    var timer: Timer?
    var index = 0
    var motionData: [[Double]] = []
    
    var frequencyScale: [Double] = []
    
    let fs = 100.0
    let windowSize = 320
    let slideDuration = 1.25
    var fftResolution = 0.0
    var slideWindowSize = 0
    var stepcount: Double = 0 {
        didSet {
            sensorDataLabel.text = String(format: "%.1f Step(s)", round(stepcount*10)/10)
        }
    }
    
    
    @IBOutlet weak var sensorDataLabel: UILabel!
    @IBOutlet weak var saveButton: UIButton!
    @IBOutlet weak var stepCountActivator: UISwitch!
    @IBOutlet weak var dataOneButton: UIButton!
    @IBOutlet weak var dataTwoButton: UIButton!
    @IBOutlet weak var dataThreeButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        for i in stride(from: 1, to: 5, by: 0.0001) {
            frequencyScale.append(i)
        }
        fftResolution = Double(fs)/Double(windowSize)
        slideWindowSize = Int(Double(fs) * slideDuration)
                
        sensorDataLabel.text = "0 Step(s)"
        saveButton.addTarget(self, action: #selector(ViewController.saveToCSV(sender:)), for: .touchDown)
        stepCountActivator.addTarget(self, action: #selector(ViewController.changeStepState(sender:)), for: .valueChanged)
        
        //
        
        dataOneButton.addTarget(self, action: #selector(ViewController.loadData(sender:)), for: .touchDown)
        dataTwoButton.addTarget(self, action: #selector(ViewController.loadData(sender:)), for: .touchDown)
        dataThreeButton.addTarget(self, action: #selector(ViewController.loadData(sender:)), for: .touchDown)
        
        
    }
    
    @objc func loadData(sender: UIButton) {
        stepcount = 0
        useCSV(fileNumber: sender.tag)
    }
    
    @objc func changeStepState(sender: UISwitch) {
        
        if sender.isOn {
            setUpAccelerometer(sampleFrequency: fs)
        } else {
            deactivateAccelerometer()
        }
    }
    
    
    @IBAction func resetCount(_ sender: UIButton) {
        stepcount = 0
    }
    
    @objc func saveToCSV(sender: UIButton) {
        let fileManager = FileManager.default
        
        var csvString = ""
        motionData.forEach { elem in
            csvString.append("\(elem[0]),\(elem[1]),\(elem[2]),\(elem[3])\n")
        }
        
        do {
            let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil, create: false)
            let fileURL = path.appendingPathComponent("data_\(Date()).csv")
            try csvString.write(to: fileURL, atomically: true, encoding: .utf8)
            print("Saved to \(fileURL.absoluteString)")
        } catch {
            print(error.localizedDescription)
        }
    }
    
    func useCSV(fileNumber: Int) {
        
        if stepCountActivator.isOn {
            stepCountActivator.setOn(false, animated: true)
        }
        
        let fileName = "data\(fileNumber)"
        if let rows = getCSV(fname: fileName) {
            var rowsMutable = rows
            rowsMutable.removeLast()
            motionData = rowsMutable
            for i in stride(from: 0, to: motionData.count-windowSize, by: slideWindowSize) {
                index = i
                print("Index: \(index)")
                doStepCounting()
            }
        }
    }
    
    func setUpAccelerometer(sampleFrequency: Double) {
        if self.motion.isAccelerometerAvailable {
            self.motion.accelerometerUpdateInterval = 1.0 / sampleFrequency
        }
        
        self.timer = Timer(fire: Date(), interval: (1.0/sampleFrequency), repeats: true, block: { [weak self] timer in
            guard let strongself = self else {
                return
            }
            if let data = strongself.motion.deviceMotion {
                strongself.motionData.append([data.timestamp, data.userAcceleration.x, data.userAcceleration.y, data.userAcceleration.z])
                
                
                if strongself.motionData.count >= strongself.windowSize && strongself.motionData.count % strongself.slideWindowSize == 0 {

                    strongself.doStepCounting()
                }
            }
        })
        
        activateAccelerometer()
    }
    
    func activateAccelerometer() {
        self.motion.startDeviceMotionUpdates()
        RunLoop.current.add(self.timer!, forMode: .default)
    }
    
    func deactivateAccelerometer() {
        self.motion.stopDeviceMotionUpdates()
        self.timer!.invalidate()
        self.motionData = []
        self.index = 0
        
    }
    
    func doStepCounting() {
        
        let axisIndex = getMostSensitiveAxis(from: index, arraySize: windowSize, data: motionData)+1
        
        var axis: [Double] = []
        axis = motionData.reduce(into: axis) { partialRes, row in
            partialRes.append(row[axisIndex])
        }
        
        guard let spectrum = computeSpectrum(axis: axis, startIndex: index, maxIndex: index+windowSize) else {
            fatalError("Could not compute spectrum")
        }
        
        let w0 = spectrum[0...1].reduce(0, +)/2
        let wc = spectrum[2...6].reduce(0, +)/6
        
        let coefficients = getCoefficients(exp: [0,1,2,3,4], spectrum: [Float](spectrum[2...6]))
        let derivative = differentiate(coefficients: coefficients)
        
        let functionVals = vDSP.evaluatePolynomial(usingCoefficients: coefficients, withVariables: frequencyScale)
        let derivativeVals = vDSP.evaluatePolynomial(usingCoefficients: derivative, withVariables: frequencyScale)
        
        let zeroPoints = derivativeVals.indices.filter{ derivativeVals[$0] >= 0 && derivativeVals[$0] <= 0.5 }
        
    
        guard let maximumIndex = zeroPoints.max(by: { functionVals[$0] < functionVals[$1] }) else {
            return
        }
        let maximumFreq = frequencyScale[maximumIndex]
        
        print("Local Maximum \(maximumFreq)")
       
        if (wc > w0) && (wc > 10) {
            let fw = fftResolution * (maximumFreq + 1.0)
            let c = slideDuration * fw
            stepcount+=c
        }
        index+=slideWindowSize
        
        
        //print(stepcount)
    }
    
    func differentiate(coefficients: [Double]) -> [Double]{
        
        var derivative: [Double] = []
        var index = 0
        for power in stride(from: coefficients.count-1, to: 0, by: -1) {
            derivative.append(coefficients[index]*Double(power))
            index+=1
        }
        
        return derivative
    }
    
    
    
    func computeSpectrum(axis: [Double], startIndex: Int, maxIndex: Int) -> [Float]? {
        
        let signal: [Float] = axis[startIndex..<maxIndex].map { Float($0) }
        
        let n = signal.count
        
        let halfN = Int(n/2)

        var forwardInputReal = [Float](repeating: 0,
                                       count: halfN)
        var forwardInputImag = [Float](repeating: 0,
                                       count: halfN)
    

        forwardInputReal.withUnsafeMutableBufferPointer { forwardInputRealPtr in
            forwardInputImag.withUnsafeMutableBufferPointer { forwardInputImagPtr in
                
                var forwardInput = DSPSplitComplex(realp: forwardInputRealPtr.baseAddress!,
                                                                           imagp: forwardInputImagPtr.baseAddress!)
                signal.withUnsafeBytes {
                    vDSP_ctoz([DSPComplex]($0.bindMemory(to: DSPComplex.self)), 2, &forwardInput, 1, vDSP_Length(halfN))
                }
            }
        }
        
        guard let dft = try? vDSP.DiscreteFourierTransform(previous: nil, count: n, direction: .forward, transformType: .complexReal, ofType: Float.self) else {
            fatalError("Could not setup dft")
        }
        
        var output = dft.transform(real: forwardInputReal, imaginary: forwardInputImag)
        


       let autospectrum = [Float](unsafeUninitializedCapacity: halfN) {
            autospectrumBuffer, initializedCount in

            // The `vDSP_zaspec` function accumulates its output. Clear the
            // uninitialized `autospectrumBuffer` before computing the spectrum.
            vDSP.clear(&autospectrumBuffer)

           output.real.withUnsafeMutableBufferPointer { forwardOutputRealPtr in
               output.imaginary.withUnsafeMutableBufferPointer { forwardOutputImagPtr in

                    var frequencyDomain = DSPSplitComplex(realp: forwardOutputRealPtr.baseAddress!,
                                                          imagp: forwardOutputImagPtr.baseAddress!)

                    
                    vDSP_zvabs(&frequencyDomain, 1, autospectrumBuffer.baseAddress!, 1, vDSP_Length(halfN))
                }
            }
            initializedCount = halfN
        }
        
        return autospectrum
    }
    
    func getCoefficients(exp: [Double], spectrum: [Float]) -> [Double] {
        let spec_sim2d: [simd_double2] = [
            simd_double2(1, Double(spectrum[0])),
            simd_double2(2, Double(spectrum[1])),
            simd_double2(3, Double(spectrum[2])),
            simd_double2(4, Double(spectrum[3])),
            simd_double2(5, Double(spectrum[4])),
            ]
        let vandermonde: [[Double]] = spec_sim2d.map { point in
            let bases = [Double](repeating: point.x, count: spec_sim2d.count)
            
            return vForce.pow(bases: bases, exponents: exp)
        }
        
        var a = vandermonde.flatMap{ $0 }
        var b = spec_sim2d.map{ $0.y }
        do {
            try ViewController.solveLinearSystem(a: &a, a_rowCount: spec_sim2d.count, a_columnCount: spec_sim2d.count, b: &b, b_count: spec_sim2d.count)
        } catch {
            fatalError("Unable to solve linear system.")
        }
        
        vDSP.reverse(&b)
        
        
        return b
        
       
        
        
        
        
    }
    
    func getMostSensitiveAxis(from startIndex: Int, arraySize: Int, data: [[Double]]) -> Int{
        
        var x: Double = 0.0
        var y: Double = 0.0
        var z: Double = 0.0
        
        for index in startIndex..<startIndex+arraySize {
            x+=abs(data[index][1])
            y+=abs(data[index][2])
            z+=abs(data[index][3])
        }
        let sizeAsFloat = Double(arraySize)
        let axesSums = [x/sizeAsFloat, y/sizeAsFloat, z/sizeAsFloat]
        
        let maxIndex = axesSums.indices.max { axesSums[$0] < axesSums[$1] }!
        return maxIndex
    }


    func getCSV(fname: String) -> [[Double]]? {
        var contents = ""
        
        guard let filePath = Bundle.main.path(forResource: fname, ofType: "csv") else {
            printContent("Invalid Filepath provided")
            return nil
        }
        do {
            contents = try String(contentsOfFile: filePath, encoding: .utf8)
        } catch {
            print("File Read Error for file \(filePath)")
            return nil
        }
        
        var rows = contents.components(separatedBy: "\n")
        rows.removeFirst()
        var values: [[Double]] = []
        for row in rows {
            let rowValues: [Double] = row.components(separatedBy: ",").map{ Double($0) ?? 0.0}
            values.append(rowValues)
        }
        
        
        
        return values
    }
}

