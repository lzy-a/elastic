/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
package app

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/scale"
	"k8s.io/klog"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"

	resourceclient "k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1"
	"k8s.io/metrics/pkg/client/custom_metrics"
	"k8s.io/metrics/pkg/client/external_metrics"
)

func startHPAController(ctx ControllerContext) (http.Handler, bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "autoscaling", Version: "v1", Resource: "horizontalpodautoscalers"}] {
		return nil, false, nil
	}

	if ctx.ComponentConfig.HPAController.HorizontalPodAutoscalerUseRESTClients {
		// use the new-style clients if support for custom metrics is enabled
		return startHPAControllerWithRESTClient(ctx)
	}

	return startHPAControllerWithLegacyClient(ctx)
}

func StartHPAController() (http.Handler, bool, error) {
	s, err := options.NewKubeControllerManagerOptions()
	if err != nil {
		klog.Fatalf("unable to initialize command options: %v", err)
	}
	c, err := s.Config(KnownControllers(), ControllersDisabledByDefault.List())
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	rootClientBuilder := controller.SimpleControllerClientBuilder{
		ClientConfig: c.Kubeconfig,
	}
	var clientBuilder controller.ControllerClientBuilder
	if c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials {
		if len(c.ComponentConfig.SAController.ServiceAccountKeyFile) == 0 {
			// It's possible another controller process is creating the tokens for us.
			// If one isn't, we'll timeout and exit when our client builder is unable to create the tokens.
			klog.Warningf("--use-service-account-credentials was specified without providing a --service-account-private-key-file")
		}

		if shouldTurnOnDynamicClient(c.Client) {
			klog.V(1).Infof("using dynamic client builder")
			//Dynamic builder will use TokenRequest feature and refresh service account token periodically
			clientBuilder = controller.NewDynamicClientBuilder(
				restclient.AnonymousClientConfig(c.Kubeconfig),
				c.Client.CoreV1(),
				"kube-system")
		} else {
			klog.V(1).Infof("using legacy client builder")
			clientBuilder = controller.SAControllerClientBuilder{
				ClientConfig:         restclient.AnonymousClientConfig(c.Kubeconfig),
				CoreClient:           c.Client.CoreV1(),
				AuthenticationClient: c.Client.AuthenticationV1(),
				Namespace:            "kube-system",
			}
		}
	} else {
		clientBuilder = rootClientBuilder
	}
	ctx := context.TODO()
	controllerContext, err := CreateControllerContext(c.Complete(), rootClientBuilder, clientBuilder, ctx.Done())
	if err != nil {
		log.Fatalf("error building controller context: %v", err)
	}

	_, started, err := startHPAController(controllerContext)
	if err != nil {
		log.Fatalf("Error starting HPA controller: %v", err)
	}

	return nil, started, err
}

func startHPAControllerWithRESTClient(ctx ControllerContext) (http.Handler, bool, error) {
	clientConfig := ctx.ClientBuilder.ConfigOrDie("horizontal-pod-autoscaler")
	hpaClient := ctx.ClientBuilder.ClientOrDie("horizontal-pod-autoscaler")

	apiVersionsGetter := custom_metrics.NewAvailableAPIsGetter(hpaClient.Discovery())
	// invalidate the discovery information roughly once per resync interval our API
	// information is *at most* two resync intervals old.
	go custom_metrics.PeriodicallyInvalidate(
		apiVersionsGetter,
		ctx.ComponentConfig.HPAController.HorizontalPodAutoscalerSyncPeriod.Duration,
		ctx.Stop)

	metricsClient := metrics.NewRESTMetricsClient(
		resourceclient.NewForConfigOrDie(clientConfig),
		custom_metrics.NewForConfig(clientConfig, ctx.RESTMapper, apiVersionsGetter),
		external_metrics.NewForConfigOrDie(clientConfig),
	)
	return startHPAControllerWithMetricsClient(ctx, metricsClient)
}

func startHPAControllerWithLegacyClient(ctx ControllerContext) (http.Handler, bool, error) {
	hpaClient := ctx.ClientBuilder.ClientOrDie("horizontal-pod-autoscaler")
	metricsClient := metrics.NewHeapsterMetricsClient(
		hpaClient,
		metrics.DefaultHeapsterNamespace,
		metrics.DefaultHeapsterScheme,
		metrics.DefaultHeapsterService,
		metrics.DefaultHeapsterPort,
	)
	return startHPAControllerWithMetricsClient(ctx, metricsClient)
}

func startHPAControllerWithMetricsClient(ctx ControllerContext, metricsClient metrics.MetricsClient) (http.Handler, bool, error) {
	hpaClient := ctx.ClientBuilder.ClientOrDie("horizontal-pod-autoscaler")
	hpaClientConfig := ctx.ClientBuilder.ConfigOrDie("horizontal-pod-autoscaler")

	// we don't use cached discovery because DiscoveryScaleKindResolver does its own caching,
	// so we want to re-fetch every time when we actually ask for it
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(hpaClient.Discovery())
	scaleClient, err := scale.NewForConfig(hpaClientConfig, ctx.RESTMapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		return nil, false, err
	}

	go podautoscaler.NewHorizontalController(
		hpaClient.CoreV1(),
		scaleClient,
		hpaClient.AutoscalingV1(),
		ctx.RESTMapper,
		metricsClient,
		ctx.InformerFactory.Autoscaling().V1().HorizontalPodAutoscalers(),
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.ComponentConfig.HPAController.HorizontalPodAutoscalerSyncPeriod.Duration,
		ctx.ComponentConfig.HPAController.HorizontalPodAutoscalerDownscaleStabilizationWindow.Duration,
		ctx.ComponentConfig.HPAController.HorizontalPodAutoscalerTolerance,
		ctx.ComponentConfig.HPAController.HorizontalPodAutoscalerCPUInitializationPeriod.Duration,
		ctx.ComponentConfig.HPAController.HorizontalPodAutoscalerInitialReadinessDelay.Duration,
	).Run(ctx.Stop)
	return nil, true, nil
}
