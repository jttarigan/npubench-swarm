// NPUBench determinism harness for iOS (study 3). One screen: a log and run
// buttons. Results go to the app's Documents folder (visible in Finder, and
// pulled with `xcrun devicectl device copy from --domain-type appDataContainer`).
// The replay input is the bundled replay_input.bin (the Tab S6 Lite CPU F32
// trace) unless a replay_input.bin was copied into Documents.

#import <UIKit/UIKit.h>
#import <sys/utsname.h>

#include <string>
#include <vector>

#include "Harness.h"

static NSString* const kDoneMark = @"=== ALL DONE";

@interface ViewController : UIViewController
@property(nonatomic, strong) UITextView* logView;
@property(nonatomic, strong) NSMutableArray<UIButton*>* buttons;
@property(nonatomic, assign) BOOL busy;
@end

@implementation ViewController

- (NSString*)docs {
    return NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
}

- (void)log:(NSString*)s {
    NSString* line = [NSString stringWithFormat:@"%@ %@\n",
                      [NSDateFormatter localizedStringFromDate:[NSDate date]
                                                     dateStyle:NSDateFormatterNoStyle
                                                     timeStyle:NSDateFormatterMediumStyle], s];
    NSString* path = [[self docs] stringByAppendingPathComponent:@"session_log.txt"];
    NSFileHandle* fh = [NSFileHandle fileHandleForWritingAtPath:path];
    if (!fh) {
        [[NSFileManager defaultManager] createFileAtPath:path contents:nil attributes:nil];
        fh = [NSFileHandle fileHandleForWritingAtPath:path];
    }
    [fh seekToEndOfFile];
    [fh writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
    [fh closeFile];
    dispatch_async(dispatch_get_main_queue(), ^{
        self.logView.text = [self.logView.text stringByAppendingString:line];
        [self.logView scrollRangeToVisible:NSMakeRange(self.logView.text.length, 0)];
    });
}

- (UIButton*)button:(NSString*)title action:(SEL)sel {
    UIButton* b = [UIButton buttonWithType:UIButtonTypeSystem];
    [b setTitle:title forState:UIControlStateNormal];
    b.titleLabel.font = [UIFont boldSystemFontOfSize:20];
    b.backgroundColor = [UIColor colorWithRed:0.1 green:0.3 blue:0.3 alpha:1];
    [b setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    b.layer.cornerRadius = 10;
    [b addTarget:self action:sel forControlEvents:UIControlEventTouchUpInside];
    [self.buttons addObject:b];
    return b;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.blackColor;
    self.buttons = [NSMutableArray array];
    UIStackView* stack = [[UIStackView alloc] initWithArrangedSubviews:@[
        [self button:@"RUN ALL  XL" action:@selector(runXL)],
        [self button:@"RUN ALL  TINY" action:@selector(runTiny)],
        [self button:@"QUICK TEST (CPU F32, 1 GAME)" action:@selector(runQuick)],
    ]];
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 12;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    self.logView = [[UITextView alloc] init];
    self.logView.editable = NO;
    self.logView.backgroundColor = UIColor.blackColor;
    self.logView.textColor = [UIColor colorWithRed:0.4 green:1 blue:0.9 alpha:1];
    self.logView.font = [UIFont monospacedSystemFontOfSize:11 weight:UIFontWeightRegular];
    self.logView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:stack];
    [self.view addSubview:self.logView];
    UILayoutGuide* g = self.view.safeAreaLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:g.topAnchor constant:12],
        [stack.leadingAnchor constraintEqualToAnchor:g.leadingAnchor constant:16],
        [stack.trailingAnchor constraintEqualToAnchor:g.trailingAnchor constant:-16],
        [stack.heightAnchor constraintEqualToConstant:3 * 52 + 2 * 12],
        [self.logView.topAnchor constraintEqualToAnchor:stack.bottomAnchor constant:12],
        [self.logView.leadingAnchor constraintEqualToAnchor:g.leadingAnchor constant:8],
        [self.logView.trailingAnchor constraintEqualToAnchor:g.trailingAnchor constant:-8],
        [self.logView.bottomAnchor constraintEqualToAnchor:g.bottomAnchor],
    ]];
    struct utsname u;
    uname(&u);
    [self log:[NSString stringWithFormat:@"NPUBench determinism harness. Device %s, iOS %@, LiteRT %s",
               u.machine, UIDevice.currentDevice.systemVersion, liteRtVersion().c_str()]];
    // started from the Mac: devicectl device process launch ... -- --autorun=quick|xl|tiny|all
    for (NSString* a in NSProcessInfo.processInfo.arguments)
        if ([a isEqualToString:@"--legacy-delegate-order"]) {
            g_legacyDelegateOrder = true;
            [self log:@"legacy delegate order (add at create, resize after)"];
        }
    for (NSString* a in NSProcessInfo.processInfo.arguments) {
        if (![a hasPrefix:@"--autorun="]) continue;
        NSString* what = [a substringFromIndex:10];
        [self log:[@"autorun " stringByAppendingString:what]];
        if ([what isEqualToString:@"quick"]) [self runQuick];
        else if ([what isEqualToString:@"xl"]) [self runXL];
        else if ([what isEqualToString:@"tiny"]) [self runTiny];
        else if ([what isEqualToString:@"all"]) [self runAll];
        else if ([what hasPrefix:@"repeat:"]) [self runRepeat:[what substringFromIndex:7]];
    }
}

// repeat:<backend>:<variant>:<prec>:<games>, e.g. repeat:gpu:tiny:int8:10
- (void)runRepeat:(NSString*)spec {
    NSArray<NSString*>* p = [spec componentsSeparatedByString:@":"];
    if (p.count != 4 || self.busy) return;
    self.busy = YES;
    HarnessConfig c;
    c.backend = [p[0] isEqualToString:@"gpu"] ? IosBackend::GPU
              : [p[0] isEqualToString:@"npu"] ? IosBackend::NPU : IosBackend::CPU;
    c.variant = p[1].UTF8String;
    c.int8 = [p[2] isEqualToString:@"int8"];
    int games = p[3].intValue;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::string models = NSBundle.mainBundle.resourcePath.UTF8String;
        std::string docs = [self docs].UTF8String;
        LogFn log = [self](const std::string& s) { [self log:@(s.c_str())]; };
        int ok = 0;
        for (int i = 0; i < games; i++) ok += runGame(c, models, docs, log);
        [self log:[NSString stringWithFormat:@"%@ (repeat %@): %d ok", kDoneMark, spec, ok]];
        self.busy = NO;
    });
}

- (void)runAll {
    if (self.busy) return;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        [self runVariant:"xlarge" games:3 quick:NO];
        while (self.busy) [NSThread sleepForTimeInterval:1.0];
        [self runVariant:"tiny" games:3 quick:NO];
    });
}

- (void)setBusy:(BOOL)busy {
    _busy = busy;   // read by runAll's wait loop
    dispatch_async(dispatch_get_main_queue(), ^{
        for (UIButton* b in self.buttons) b.enabled = !busy;
        UIApplication.sharedApplication.idleTimerDisabled = busy;
    });
}

- (void)runXL { [self runVariant:"xlarge" games:3 quick:NO]; }
- (void)runTiny { [self runVariant:"tiny" games:3 quick:NO]; }
- (void)runQuick { [self runVariant:"xlarge" games:1 quick:YES]; }

// Every config: `games` free games, then a replay of the reference trace.
- (void)runVariant:(const char*)variant games:(int)games quick:(BOOL)quick {
    if (self.busy) return;
    self.busy = YES;
    std::string variantStr = variant;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::string models = NSBundle.mainBundle.resourcePath.UTF8String;
        std::string docs = [self docs].UTF8String;
        NSString* docInput = [[self docs] stringByAppendingPathComponent:@"replay_input.bin"];
        NSString* replayPath = [NSFileManager.defaultManager fileExistsAtPath:docInput]
            ? docInput
            : [NSBundle.mainBundle pathForResource:@"replay_input" ofType:@"bin"];
        LogFn log = [self](const std::string& s) { [self log:@(s.c_str())]; };
        std::vector<HarnessConfig> cfgs;
        for (IosBackend b : {IosBackend::CPU, IosBackend::GPU, IosBackend::NPU})
            for (bool q : {false, true}) {
                if (b == IosBackend::NPU && q) continue;   // Core ML delegate: float models only
                cfgs.push_back({b, q, variantStr});
            }
        if (quick) cfgs.resize(1);
        int ok = 0, failed = 0;
        for (const HarnessConfig& c : cfgs) {
            [self log:[NSString stringWithFormat:@"--- %s %s %s", backendTag(c.backend),
                       c.variant.c_str(), c.int8 ? "int8" : "f32"]];
            for (int i = 0; i < games; i++)
                (runGame(c, models, docs, log) ? ok : failed)++;
            if (!quick && replayPath && c.variant == "xlarge")
                (runReplay(c, models, replayPath.UTF8String, docs, log) ? ok : failed)++;
        }
        [self log:[NSString stringWithFormat:@"%@ (%s): %d ok, %d failed", kDoneMark, variant,
                   ok, failed]];
        self.busy = NO;
    });
}

@end

@interface SceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(nonatomic, strong) UIWindow* window;
@end

@implementation SceneDelegate
- (void)scene:(UIScene*)scene willConnectToSession:(UISceneSession*)session
      options:(UISceneConnectionOptions*)options {
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene*)scene];
    self.window.rootViewController = [[ViewController alloc] init];
    [self.window makeKeyAndVisible];
}
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation AppDelegate
- (UISceneConfiguration*)application:(UIApplication*)app
    configurationForConnectingSceneSession:(UISceneSession*)session
                                   options:(UISceneConnectionOptions*)options {
    UISceneConfiguration* c = [[UISceneConfiguration alloc] initWithName:@"Default"
                                                             sessionRole:session.role];
    c.delegateClass = SceneDelegate.class;
    return c;
}
@end

int main(int argc, char* argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass(AppDelegate.class));
    }
}
