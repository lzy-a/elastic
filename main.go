package main

import (
	"log"

	app "k8s.io/kubernetes/cmd/kube-controller-manager/app" // 请替换成实际的包路径
)

func main() {

	_, started, err := app.StartHPAController()
	if err != nil {
		log.Fatalf("Error starting HPA controller: %v", err)
	}

	if !started {
		log.Fatalf("HPA controller not started")
	}
	select {}
}
