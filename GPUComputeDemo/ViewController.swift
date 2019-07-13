//
//  ViewController.swift
//  GPUComputeDemo
//
//  Created by liuRuiLong on 2019/7/11.
//  Copyright © 2019 Ray. All rights reserved.
//

import UIKit
import Metal
import MetalKit

public class MetalHelper: NSObject {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let textureLoader: MTKTextureLoader
    static let shared: MetalHelper = MetalHelper.init()
    
    private override init(){
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
        textureLoader = MTKTextureLoader.init(device: device)
        super.init()
    }
}

class ViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    let metalHelper: MetalHelper = MetalHelper.shared
    var outputTextureDesc: MTLTextureDescriptor!
    var inputTexture: MTLTexture!
    var pipline: MTLComputePipelineState?
    var originImage: UIImage!
    var currentRadius: Int = 0
    var weightMap: [Int : Float] = [:]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        outputTextureDesc = getOuputTextureDesc()
        
        let lib = metalHelper.device.makeDefaultLibrary()
        if let function = lib?.makeFunction(name: "blur_kernel") {
            pipline = try? metalHelper.device.makeComputePipelineState(function: function)
        }
        
        if let image = imageView.image?.cgImage {
            inputTexture = try! metalHelper.textureLoader.newTexture(cgImage: image, options: [:])
        } else {
            print("还没图片")
        }
        originImage = imageView.image
        
        // 设置高斯模糊权重和
        setWeightMap()
    }
    
    @IBAction func sliderValueChanged(_ sender: Any) {
        guard let inPipline = pipline else {
            print(" pipline not ready")
            return
        }
        
        if let slider = sender as? UISlider {
            var radius = Int(slider.value + 0.5)
            
//            if radius == 0 {
//                self.imageView.image = originImage
//                return;
//            }
            
            print(radius)
            
            if currentRadius == radius {
                return;
            }
            currentRadius = radius
            
            let outputTexture = metalHelper.device.makeTexture(descriptor: outputTextureDesc)
            let commandBuffer = metalHelper.queue.makeCommandBuffer()
            let encoder = commandBuffer?.makeComputeCommandEncoder()
            var sumOfWeight = weightMap[radius];
            encoder?.setTexture(inputTexture, index: 0)
            encoder?.setTexture(outputTexture, index: 1)
            encoder?.setBytes(&radius, length: MemoryLayout<Int>.size, index: 0)
            encoder?.setBytes(&sumOfWeight, length: MemoryLayout<Float>.size, index: 1)

            encoder?.setComputePipelineState(inPipline)
            
            let width = inPipline.threadExecutionWidth
            let height = inPipline.maxTotalThreadsPerThreadgroup / width
            let threadsPerGroup = MTLSize.init(width: width, height: height, depth: 1)
            
            let slices = (outputTexture!.arrayLength * 4 + 3)/4
            let groupWidth = (outputTexture!.width + width - 1)/width
            let groupHeight = (outputTexture!.height + height - 1)/height
            let groups = MTLSize.init(width: groupWidth, height: groupHeight, depth: slices)
            encoder?.setComputePipelineState(inPipline)
            encoder?.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
            encoder?.endEncoding()
            
            let date = Date.init()
            
            commandBuffer?.addCompletedHandler({ (buffer) in
                let ciimage = CIImage.init(mtlTexture: outputTexture!, options: [:])
                let uiimage = UIImage.init(ciImage: ciimage!, scale: 1.0,orientation: .up)
                print(uiimage)
                DispatchQueue.main.async {
                    self.imageView.image = uiimage;
                }
                
                print(Date.init().timeIntervalSince(date))
            })
            
            commandBuffer?.commit()
        }
    }
    
    /// 获取输出 texture 描述
    ///
    /// - Returns: 输出 texture 描述
    func getOuputTextureDesc() -> MTLTextureDescriptor {
        let textureDes = MTLTextureDescriptor.init()
        textureDes.textureType = .type2D
        textureDes.width = 658
        textureDes.height = 987
        textureDes.depth = 1
        textureDes.pixelFormat = .rgba32Float
        textureDes.usage = [.shaderWrite, .shaderRead]
        textureDes.storageMode = .shared
        return textureDes
    }
    
    /// 设置高斯模糊权重
    func setWeightMap() {
        let radius = 10;
        let m = 5;
        for r in 1...radius {
            let r_min = 0 - r;
            let r_max = 0 + r + 1;
            var sum_of_weight: Float = 0.0
            for i in r_min..<r_max {
                for j in r_min..<r_max {
                    let weight = 1/((2 * 3.1415926) * pow(Double(m), 2.0)) * exp(-((pow(Double(i), 2.0) + pow(Double(j), 2.0))/(2 * pow(Double(m), 2.0))))
                    sum_of_weight += Float(weight)
                }
            }
            weightMap[r] = sum_of_weight
        }
    }
}

