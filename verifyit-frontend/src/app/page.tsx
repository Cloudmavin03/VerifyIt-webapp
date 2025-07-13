'use client'

import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Shield, AlertTriangle, BarChart3, BookOpen, Phone, Search, TrendingUp, Users, CheckCircle, Loader } from "lucide-react"

interface AnalysisResult {
  prediction: string
  probability: number
  risk_level: string
  confidence: string
  indicators: string[]
}

interface Stats {
  total_scans: number
  scams_detected: number
  accuracy_rate: number
  users_protected: number
}

interface EmergencyContact {
  name: string
  description: string
  phone: string
  website: string
}

export default function VerifyIt() {
  const [analysisText, setAnalysisText] = useState('')
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [stats, setStats] = useState<Stats | null>(null)
  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([])

  // Fetch stats on component mount
  useEffect(() => {
    fetchStats()
    fetchEmergencyContacts()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/stats')
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const fetchEmergencyContacts = async () => {
    try {
      const response = await fetch('http://localhost:8000/emergency-contacts')
      const data = await response.json()
      setEmergencyContacts(data.contacts)
    } catch (error) {
      console.error('Failed to fetch emergency contacts:', error)
    }
  }

  const analyzeText = async () => {
    if (!analysisText.trim()) return

    setIsAnalyzing(true)
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: analysisText }),
      })
      
      if (response.ok) {
        const result = await response.json()
        setAnalysisResult(result)
      } else {
        console.error('Analysis failed')
      }
    } catch (error) {
      console.error('Error analyzing text:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 rounded-lg bg-green-600 flex items-center justify-center">
                <Shield className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">VerifyIt Nigeria</h1>
                <p className="text-sm text-gray-500">AI-Powered Investment Scam Detection</p>
              </div>
            </div>
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
              🇳🇬 Nigerian Context
            </Badge>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <Tabs defaultValue="dashboard" className="w-full">
          {/* Navigation Tabs */}
          <TabsList className="grid w-full grid-cols-5 mb-8">
            <TabsTrigger value="dashboard" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="detection" className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Scan Message
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="education" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Education
            </TabsTrigger>
            <TabsTrigger value="report" className="flex items-center gap-2">
              <Phone className="h-4 w-4" />
              Report Scam
            </TabsTrigger>
          </TabsList>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Scans</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.total_scans?.toLocaleString() || '1,247'}</div>
                  <p className="text-xs text-muted-foreground">+12 from yesterday</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Scams Detected</CardTitle>
                  <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.scams_detected || '89'}</div>
                  <p className="text-xs text-muted-foreground">+3 from yesterday</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Accuracy Rate</CardTitle>
                  <CheckCircle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.accuracy_rate || '94.2'}%</div>
                  <p className="text-xs text-muted-foreground">+2.1% this month</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Users Protected</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.users_protected?.toLocaleString() || '2,841'}</div>
                  <p className="text-xs text-muted-foreground">+15% this month</p>
                </CardContent>
              </Card>
            </div>

            {/* Recent Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Recent Detections</CardTitle>
                  <CardDescription>Latest scam detection results</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Badge variant="destructive">High Risk</Badge>
                      <span className="text-sm">Forex trading scheme</span>
                    </div>
                    <span className="text-xs text-muted-foreground">2 min ago</span>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Badge variant="outline">Safe</Badge>
                      <span className="text-sm">Investment consultation</span>
                    </div>
                    <span className="text-xs text-muted-foreground">1 hour ago</span>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Badge variant="destructive">High Risk</Badge>
                      <span className="text-sm">MMM revival scheme</span>
                    </div>
                    <span className="text-xs text-muted-foreground">3 hours ago</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Common Scam Types</CardTitle>
                  <CardDescription>Most frequent scams in Nigeria</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Forex Trading Scams</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 h-2 bg-gray-200 rounded-full">
                          <div className="w-20 h-2 bg-red-500 rounded-full"></div>
                        </div>
                        <span className="text-xs text-muted-foreground">35%</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Ponzi Schemes</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 h-2 bg-gray-200 rounded-full">
                          <div className="w-16 h-2 bg-orange-500 rounded-full"></div>
                        </div>
                        <span className="text-xs text-muted-foreground">28%</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Crypto Doublers</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 h-2 bg-gray-200 rounded-full">
                          <div className="w-12 h-2 bg-yellow-500 rounded-full"></div>
                        </div>
                        <span className="text-xs text-muted-foreground">22%</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Other Scams</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 h-2 bg-gray-200 rounded-full">
                          <div className="w-8 h-2 bg-blue-500 rounded-full"></div>
                        </div>
                        <span className="text-xs text-muted-foreground">15%</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Scam Detection Tab */}
          <TabsContent value="detection" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Analyze Investment Message</CardTitle>
                <CardDescription>
                  Paste any investment offer, WhatsApp message, or email to check for scam indicators
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={analysisText}
                  onChange={(e) => setAnalysisText(e.target.value)}
                  placeholder="Example: Join our forex trading platform and earn ₦500,000 monthly with guaranteed returns..."
                  className="min-h-[120px]"
                />
                <Button 
                  onClick={analyzeText}
                  disabled={isAnalyzing || !analysisText.trim()}
                  className="w-full bg-green-600 hover:bg-green-700"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4 mr-2" />
                      Analyze for Scams
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Analysis Result */}
            {analysisResult && (
              <Alert className={analysisResult.prediction === 'SCAM' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                {analysisResult.prediction === 'SCAM' ? (
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                ) : (
                  <CheckCircle className="h-4 w-4 text-green-600" />
                )}
                <AlertDescription className={analysisResult.prediction === 'SCAM' ? 'text-red-800' : 'text-green-800'}>
                  <strong>
                    {analysisResult.prediction === 'SCAM' 
                      ? `High Risk Detected (${(analysisResult.probability * 100).toFixed(1)}% confidence)`
                      : `Appears Legitimate (${((1 - analysisResult.probability) * 100).toFixed(1)}% confidence)`
                    }
                  </strong><br />
                  {analysisResult.prediction === 'SCAM' ? (
                    <>
                      This message contains multiple Nigerian scam indicators including:{' '}
                      {analysisResult.indicators.slice(0, 3).join(', ')}.
                    </>
                  ) : (
                    'This message does not contain significant scam indicators, but always verify independently.'
                  )}
                </AlertDescription>
              </Alert>
            )}
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Detection Accuracy</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-green-600">{stats?.accuracy_rate || '94.2'}%</div>
                  <p className="text-sm text-muted-foreground">+2.1% from last month</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>False Positives</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-blue-600">3.1%</div>
                  <p className="text-sm text-muted-foreground">-0.5% from last month</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>User Feedback</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-purple-600">87%</div>
                  <p className="text-sm text-muted-foreground">+5% from last month</p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Education Tab */}
          <TabsContent value="education" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>🇳🇬 Nigerian Investment Scams</CardTitle>
                  <CardDescription>Common scam types targeting Nigerians</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="p-3 border rounded-lg">
                      <h4 className="font-semibold text-red-600">Forex Trading Scams</h4>
                      <p className="text-sm text-muted-foreground">Promise guaranteed daily returns, require upfront payments</p>
                    </div>
                    <div className="p-3 border rounded-lg">
                      <h4 className="font-semibold text-orange-600">Ponzi Schemes (MMM)</h4>
                      <p className="text-sm text-muted-foreground">Promise 30% monthly returns, require recruiting others</p>
                    </div>
                    <div className="p-3 border rounded-lg">
                      <h4 className="font-semibold text-yellow-600">Crypto Doublers</h4>
                      <p className="text-sm text-muted-foreground">Claim to double Bitcoin investments, use fake testimonials</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>⚠️ Red Flags to Watch</CardTitle>
                  <CardDescription>Warning signs of investment scams</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      <span>Guaranteed returns with "no risk"</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      <span>Pressure to "act now" or "limited time"</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      <span>Requires recruiting friends/family</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      <span>Unrealistic profit promises (&gt;5% monthly)</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      <span>Only communicates via WhatsApp/Telegram</span>
                    </li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Report Tab */}
          <TabsContent value="report" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>🚨 Report Investment Scams</CardTitle>
                <CardDescription>Help protect other Nigerians by reporting scams to authorities</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {emergencyContacts.map((contact, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <h4 className="font-semibold mb-2">{contact.name}</h4>
                      <p className="text-sm text-muted-foreground mb-2">{contact.description}</p>
                      <p className="text-sm font-medium">📞 {contact.phone}</p>
                      {contact.website && <p className="text-sm font-medium">🌐 {contact.website}</p>}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <footer className="border-t bg-gray-50/50 mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div>
              <p className="font-medium">🇳🇬 VerifyIt Nigeria - Protecting Nigerian investors from fraudulent schemes</p>
              <p className="mt-1">
                Research by <strong>Ononneobazi Aquah</strong> (Computer Science Student) 
                | Supervised by <strong>Prof. Moses Adah Agana</strong>
              </p>
            </div>
            <p>University of Calabar, Nigeria</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
