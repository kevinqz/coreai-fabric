// swift-tools-version:6.0
import PackageDescription

// Gate C — runtime loadability verifier. Loads + runs a `.aimodel` through the
// REAL Core AI Swift runtime and emits a JSON verdict (loads/runs + OS build).
// Build on a Mac with the target Xcode/SDK (e.g. Xcode 27); the produced verdict
// certifies the artifact against THAT runtime — the guarantee fabric's Python
// Gate B (numerical parity on the coreai-core wheel) cannot give.
let package = Package(
    name: "coreai-runtime-verify",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(url: "https://github.com/john-rocky/coreai-models", exact: "0.1.2-zoo"),
    ],
    targets: [
        .executableTarget(
            name: "coreai-runtime-verify",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models"),
            ],
            path: "Sources/coreai-runtime-verify"
        ),
    ]
)
