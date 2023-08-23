//
//  ViewController.swift
//  Assignment2
//
//  Created by Benedikt Langer on 06.12.21.
//

import UIKit
import CoreML
import CoreMotion
import simd

class ViewController: UIViewController {
    
    let motion = CMMotionManager()
    var timer: Timer?
    var motionData: [[Double]] = []
    var preProcessedData: [[Double]] = [] {
        didSet {
            if let data = preProcessedData.last {
                predict(data: data)
            }
        }
    }
    var tree: DecisionTree!
    var predictionData: [DecisionTreeOutput] = []
    
    @IBOutlet weak var predictionTableView: UITableView!
    @IBOutlet weak var liveMeasure: UISwitch!
    
    @IBOutlet weak var recordButton: UIButton!
    @IBOutlet weak var stopRecordButton: UIButton!
    @IBOutlet weak var startMeasurement: UIButton!
    @IBOutlet weak var stopMeasurement: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        predictionTableView.delegate = self
        predictionTableView.dataSource = self
        // Do any additional setup after loading the view.
        
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        
        
        guard let tree = try? DecisionTree(configuration: config) else {
            fatalError("Could not load model")
        }
        
        self.tree = tree
        
        liveMeasure.addTarget(self, action: #selector(ViewController.setUpLiveRecording(sender:)), for: .valueChanged)
        
        recordButton.addTarget(self, action: #selector(ViewController.startMeasurement(sender:)), for: .touchDown)
        startMeasurement.addTarget(self, action: #selector(ViewController.startMeasurement(sender:)), for: .touchDown)
        stopMeasurement.addTarget(self, action: #selector(ViewController.stopMeasurement(sender:)), for: .touchDown)
        stopRecordButton.addTarget(self, action: #selector(ViewController.stopAndPredict(sender:)), for: .touchDown)
        
        // for usage with a csv file use the call provided below
        // file must be in the format <Timestamp, x, y, z>
        //useCSV(filename: "<insert filename>")
        
    }
    @IBAction func resetButton(_ sender: UIButton) {
        resetData()
    }
    
    @objc func setUpLiveRecording(sender: UISwitch) {
        
        recordButton.isEnabled = !sender.isOn
        startMeasurement.isEnabled = sender.isOn
    }
    
    @objc func startMeasurement(sender: UIButton) {
        setUpAccelerometer(sampleFrequency: 20)
        liveMeasure.isEnabled = false
        sender.isEnabled = false
        if liveMeasure.isOn {
            stopMeasurement.isEnabled = true
        } else {
            stopRecordButton.isEnabled = true
        }
        activateAccelerometer()
    }
    
    @objc func stopMeasurement(sender: UIButton) {
        liveMeasure.isEnabled = true
        sender.isEnabled = false
        startMeasurement.isEnabled = true
        deactivateAccelerometer()
    }
    
    @objc func stopAndPredict(sender: UIButton) {
        liveMeasure.isEnabled = true
        sender.isEnabled = false
        recordButton.isEnabled = true
        deactivateAccelerometer()
        predictRecordedData()
    }
    
    func resetData() {
        motionData = []
        preProcessedData = []
        predictionData = []
        predictionTableView.reloadData()
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
                
                print([data.timestamp, data.userAcceleration.x, data.userAcceleration.y, data.userAcceleration.z])
                
                if strongself.liveMeasure.isOn {
                    if strongself.motionData.count >= 200 && strongself.motionData.count % 200 == 0 {

                        strongself.doDataProcess()
                    }
                }
            }
        })
        
        activateAccelerometer()
        
    }
    
    func predictRecordedData() {
        let windowAmount = motionData.count/200
        
        for i in stride(from: 0, to: windowAmount*200, by: 200) {
            let dataSlice = Array(motionData[i...i+199])
            process(dataSlice: dataSlice)
        }
        
    }
    
    func activateAccelerometer() {
        self.motion.startDeviceMotionUpdates()
        RunLoop.current.add(self.timer!, forMode: .default)
    }
    
    func deactivateAccelerometer() {
        self.motion.stopDeviceMotionUpdates()
        self.timer!.invalidate()
    }
    
    fileprivate func process(dataSlice: [[Double]]) {
        let avgAxes = (1...3).map{ dataSlice.columnMean($0) }
        let stdAxes = (1...3).map{ dataSlice.colStd($0, columnMean: avgAxes[$0-1])}
        
        let preProcessed = avgAxes + stdAxes
        preProcessedData.append(preProcessed)
    }
    
    func doDataProcess() {
        
        print("Do Preprocessing")
        let firstIndex = motionData.count - 200
        let dataSlice = Array(motionData[firstIndex..<motionData.count])
        process(dataSlice: dataSlice)
    }

    func predict(data: [Double]) {
        
        do {
            let outcome = try tree.prediction(x_avg: data[0], y_avg: data[1], z_avg: data[2], x_std: data[3], y_std: data[4], z_std: data[5])
            print(outcome.classProbability)
            predictionData.append(outcome)
            predictionTableView.reloadData()
        } catch {
            fatalError(error.localizedDescription)
        }
        
    }
    
    
    
    func useCSV(filename: String) {
        
        
        let fileName = filename
        if let rows = getCSV(fname: fileName) {
            var rowsMutable = rows
            rowsMutable.removeLast()
            motionData = rowsMutable
            
            predictRecordedData()
        }
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

extension ViewController: UITableViewDelegate, UITableViewDataSource {
    
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        predictionData.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        let cell = tableView.dequeueReusableCell(withIdentifier: "PredictionCell")!
        
        let prediction = predictionData[indexPath.row]
        let sorted = prediction.classProbability.sorted {
            $0.value > $1.value
        }
        let firstElem = sorted.first!
        
        cell.textLabel?.text = "\(firstElem.key): \(firstElem.value)"
        cell.detailTextLabel?.text = "Sample \(indexPath.row + 1)"
        
        return cell
    }
    
    
}




extension Array where Element == [Double] {
    
    
    func columnMean(_ ix1: Int) -> Double {
        
        let sum = reduce(0.0) { $0 + $1[ix1] }
        
        return sum / Double(self.count)
        
    }
    
    func colStd(_ ix1: Int, columnMean: Double) -> Double {
        let v = reduce(0) { $0 + pow(($1[ix1] - columnMean), 2.0) }
        return sqrt(v / Double(self.count-1))
    }
    
}

